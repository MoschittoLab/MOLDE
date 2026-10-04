##############################################################################
# MOLDE (Molecular Library Development & Editing) - combinatorial chemical library generator (PyQt6 GUI)
#
#   1. CONFIGURATION / CHEMISTRY DEFINITIONS
#      Three central dictionaries define much of MOLDE's chemistry and
#      filtering behavior:
#
#        reactions_dictionary
#          - Defines the available synthetic transformations and molecular
#            editing operations, their reaction type, and (where applicable)
#            RDKit reaction SMARTS.
#          - Most transformations are dispatched to generic reaction engines;
#            transformations requiring specialized logic (e.g. Ugi) implement
#            their own multi-step workflows.
#
#        filters_dictionary
#          - Defines preset physicochemical/drug-likeness criteria used by the
#            property-filtration system (Lipinski, Ghose, Veber, Egan,
#            Muegge, Rule-of-Three, etc.).
#
#        substructure_dictionary
#          - Defines SMARTS-based structural filters used to identify or
#            exclude unwanted molecular features.
#          - Contains individual functional-group/substructure filters as well
#            as established medicinal-chemistry structural-alert collections,
#            including PAINS, Brenk, NIH, and ChEMBL filters.
#
#   2. CHEMINFORMATICS / CHEMISTRY LAYER
#      RDKit-based helper functions provide the core molecular operations:
#
#        - SMILES/SDF parsing, validation, sanitization, and serialization
#        - reaction execution and combinatorial product enumeration
#        - duplicate removal / canonical-SMILES handling
#        - stereoisomer enumeration
#        - molecular-property and descriptor calculation
#        - SAScore estimation
#        - substructure/SMARTS matching and structural-alert detection
#        - molecular similarity calculations
#        - library splitting, merging, and related editing operations
#
#      Reaction execution is organized around reusable reaction-engine classes
#      (including one-component, two-component, multi-pool, and intramolecular
#      workflows), while exceptional transformations can define specialized
#      algorithms where generic SMARTS execution is insufficient.
#
#   3. PARALLELISM / BACKGROUND EXECUTION
#      Computationally expensive RDKit operations are parallelized where
#      appropriate using concurrent.futures / multiprocessing and worker
#      processes.
#
#      Reaction workers:
#        1. Resolve reactant molecules and cached SMARTS/reaction objects.
#        2. Execute RDKit ChemicalReaction.RunReactants().
#        3. Validate, sanitize, and serialize generated products.
#        4. Return compact product representations for aggregation and
#           deduplication by the parent process.
#
#      Qt worker threads / QThread subclasses coordinate long-running jobs so
#      expensive chemistry, filtration, loading, and processing operations do
#      not block the GUI event loop. Process-based parallelism performs the
#      CPU-heavy chemistry; Qt threads primarily manage asynchronous execution,
#      progress reporting, and communication with the interface.
#
#   4. FILTRATION / LIBRARY-SELECTION PIPELINE
#      Generated or imported libraries can be progressively reduced using
#      independent and composable filters:
#
#        - physicochemical/property filters
#        - predefined drug-likeness rules
#        - user-defined substructure inclusion/exclusion
#        - PAINS / Brenk / NIH / ChEMBL structural alerts
#        - SAScore-based synthetic-accessibility filtering/guidance
#        - molecular-similarity filtering
#
#      Filters operate as a pipeline rather than being coupled to library
#      generation, allowing MOLDE to process both internally generated and
#      externally imported compound collections.
#
#   5. GUI / APPLICATION LAYER
#      PyQt6 windows, dialogs, viewers, and worker-controller classes expose
#      the underlying chemistry through the graphical workflow:
#
#        import/select starting structures
#             -> choose transformation
#             -> configure reactants/options
#             -> generate/edit library
#             -> inspect products
#             -> calculate/filter properties
#             -> apply structural/similarity filters
#             -> split/merge libraries as required
#             -> export the resulting collection
#
#      The interface also manages progress reporting, logs, molecule previews,
#      user selections, cancellation/error handling, and communication between
#      background workers and the main Qt event loop.
#
#   6. DATA / STATE MANAGEMENT
#      MOLDE currently uses a mixture of object-owned state and module-level
#      shared state. Library contents, selections, reaction settings, cached
#      molecular data, and GUI state may therefore be stored either on the
#      responsible class instance or in module-level variables accessed with
#      `global`.
##############################################################################

import sys
import concurrent.futures      
import importlib  
from pathlib import Path
import urllib.request
import csv        
import io                    
import time                  
import math                  
import os                    
import multiprocessing as mp 
import pandas as pd    
import numpy as np  
from sklearn.decomposition import PCA    
from rdkit import Chem       
from rdkit import DataStructs
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, FilterCatalog #, Draw
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions  
from rdkit.Chem import rdFingerprintGenerator

# --- Qt imports --------------------------------------------------------------
from PyQt6 import QtWidgets
from PyQt6.QtGui import *
from PyQt6.QtCore import *
from PyQt6.QtWidgets import *
from PyQt6.QtSvgWidgets import QSvgWidget 
from PyQt6.QtSvg import QSvgRenderer


def ensure_known_aggregators_file():

    known_aggregators_file_path = Path(__file__).resolve().parent / "known_aggregators.smi"
    url = ("https://huggingface.co/datasets/maomlab/""AggregatorAdvisor/raw/main/raw_data.csv")

    if (known_aggregators_file_path.is_file() and known_aggregators_file_path.stat().st_size > 0):
        return known_aggregators_file_path

    print("Known aggregator database not found. Downloading...")

    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            text = resp.read().decode("utf-8-sig")

        reader = csv.DictReader(io.StringIO(text))

        with known_aggregators_file_path.open("w", encoding="utf-8") as out:

            out.write("# Known aggregators - Irwin et al., ""J Med Chem 2015, 58, 7076-7087\n")
            out.write("# Source: ""https://huggingface.co/datasets/maomlab/""AggregatorAdvisor (MIT license)\n")
            n_written = 0
            for row in reader:
                smi = row.get("smiles", "").strip()
                if smi:
                    out.write(smi + "\n")
                    n_written += 1
        print(f"Wrote {n_written:,} known aggregators to "f"{known_aggregators_file_path}")

        return known_aggregators_file_path

    except Exception as e:
        print(f"Could not download known aggregator database: {e}")

        if known_aggregators_file_path.exists():
            try:
                known_aggregators_file_path.unlink()
            except OSError:
                pass

        return None

# --- Morgan Fingerprints ---
_MORGAN_GENERATOR = (rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048))

# --- Known Aggregators ---
_KNOWN_AGGREGATORS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "known_aggregators.smi")

# --- Chemical-space plotting (matplotlib) ---
try:
    import matplotlib
    matplotlib.use("QtAgg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    MATPLOTLIB_AVAILABLE = True
    MATPLOTLIB_IMPORT_ERROR = ""
except Exception as _mpl_exc:
    import traceback
    MATPLOTLIB_AVAILABLE = False
    MATPLOTLIB_IMPORT_ERROR = f"{type(_mpl_exc).__name__}: {_mpl_exc}"
    print(
        "WARNING: matplotlib could not be imported - the 'Plot Chemical Space' "
        "feature will be disabled. Install it with: pip install matplotlib\n"
        + traceback.format_exc()
    )

# --- Synthetic Accessibility (SAScore) scorer discovery ---------------------
_sa_scorer_module = None                       
SASCORE_SOURCE = "unavailable"  

possible_imports = [
    "rdkit.Chem.Contrib.SA_Score.sascorer",
    "rdkit.Chem.SA_Score.sascorer",
    "sascorer"
]

try:
    from rdkit.Chem import RDConfig
    contrib_sa_dir = os.path.join(RDConfig.RDContribDir, 'SA_Score')
    if os.path.isdir(contrib_sa_dir) and contrib_sa_dir not in sys.path:
        sys.path.append(contrib_sa_dir)
except Exception:
    pass

_bundled_sascorer_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sascorer_bundle")
if os.path.isdir(_bundled_sascorer_dir) and _bundled_sascorer_dir not in sys.path:
    sys.path.insert(0, _bundled_sascorer_dir) 

for module_name in possible_imports:
    try:
        _sa_scorer_module = importlib.import_module(module_name)
        SASCORE_SOURCE = f"rdkit-contrib-sascorer ({module_name})"
        break
    except Exception:
        continue

if _sa_scorer_module is None:
    print(
        "WARNING: Could not load the RDKit/Ertl-Schuffenhauer SA_Score "
        "implementation. SAScore calculations will be unavailable."
    )

# --- PAINS / BRENK / NIH / ChEMBL-alert Filter Catalogues set up ---------------------
_pains_catalog = None
def _get_pains_catalog():
    global _pains_catalog
    if _pains_catalog is None:
        params = FilterCatalog.FilterCatalogParams()
        params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_A)
        params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_B)
        params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_C)
        _pains_catalog = FilterCatalog.FilterCatalog(params)
    return _pains_catalog

_brenk_catalog = None
def _get_brenk_catalog():
    global _brenk_catalog
    if _brenk_catalog is None:
        params = FilterCatalog.FilterCatalogParams()
        params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)
        _brenk_catalog = FilterCatalog.FilterCatalog(params)
    return _brenk_catalog

_nih_catalog = None
def _get_nih_catalog():
    global _nih_catalog
    if _nih_catalog is None:
        params = FilterCatalog.FilterCatalogParams()
        params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.NIH)
        _nih_catalog = FilterCatalog.FilterCatalog(params)
    return _nih_catalog

_chembl_catalog = None
def _get_chembl_catalog():
    global _chembl_catalog
    if _chembl_catalog is None:
        params = FilterCatalog.FilterCatalogParams()
        for cat in ("CHEMBL_Glaxo", "CHEMBL_Dundee", "CHEMBL_BMS", "CHEMBL_SureChEMBL",
                    "CHEMBL_MLSMR", "CHEMBL_Inpharmatica", "CHEMBL_LINT"):
            params.AddCatalog(getattr(FilterCatalog.FilterCatalogParams.FilterCatalogs, cat))
        _chembl_catalog = FilterCatalog.FilterCatalog(params)
    return _chembl_catalog


# --------------- Reactions Dictionary ---------------------
reactions_dictionary = {
    "Ugi Reaction": {
        "smarts": "", # unused placeholder - the Ugi's 3 sequential SMARTS steps are inside UgiReaction.run() instead
        "type": "multi",
        "A": "Not App.",
        "B": "Not App."
    },
    "Amide to Sulfonamide": {
        "smarts": "[*:1][#6](=O)[#7:2] >> [*:1]S(=O)(=O)[#7:2]",
        "type": "one",
        "Starting Material": "Amide",
        "Produc": "Sulfonamide"
    },
    "Amide Coupling": {#"[C:1](=[O:2])O.[Nh!$(NC=O):3] >> [C:1](=[O:2])[N:3]"
        "smarts": {
            "1": "[C:1](=[O:2])O",
            "2": "[Nh!$(N*=O):3]",
            "3": "[C:1](=[O:2])[N:3]"
        },
        "type": "two",
        "A": "-COOH",
        "B": "-NH"
    },
    "Amide Flip": {
        "smarts": "[#6:1][CX3:4](=O)[#7:5]([*:2])[*:3] >> [*:2][CX3:4](=O)[#7:5]([#6:1])[*:3]",
        "type": "one",
        "Starting Material": "C1-CON-*2",
        "Product": "*2-CON-C1"
    },
    "Amide UNcoupling": {
        "smarts": "[C](=O)[N:3] >> [N:3]",
        "type": "one",
        "Starting Material": "Amide",
        "Product": "HN"
    },
    "Sonogashira Coupling": {
        "smarts": {#"[#6:1]#[#6:2]-[#1:3].[Cl,Br,I,OS(=O)(=O)C(F)(F)F](-[#6:4] >> [#6:1]#[#6:2]-[#6:4]"
            "1": "[#6:1]#[#6:2]-[#1:3]",
            "2": "[Cl,Br,I,$(OS(=O)(=O)C(F)(F)F)](-[#6:4])",
            "3": "[#6:1]#[#6:2]-[#6:4]"
        },
        "type": "two",
        "A": "Terminal Alkyne",
        "B": "Aryl/Vinyl Halide/Triflate"
    },
    "Suzuki-Miyaura Coupling": {
        "smarts": {#"B([#6:1])(O)O.[#17,#35,#53]([c:2]) >> [#6:1]-[c:2]",
            "1": "B([#6:1])(O)O",
            "2": "[#17,#35,#53]([c:2])",
            "3": "[#6:1]-[c:2]"
        },
        "type": "two",
        "A": "Boronate",
        "B": "Aryl Halide"
    },
    "Miyaura Borylation": {
        "smarts": "[#6:1][Br,I]>>[#6:1]B1OC(C)(C)C(C)(C)O1",
        "type" : "one",
        "Starting Material": "Bromide or Iodide",
        "Product": "Bpin Boronate"
    },
    "Chan Lam reaction": {
        "smarts": {
            "1": "[c:1]B(O)O",
            "2": "[#7,#8,#16;H1,H2:2]",
            "3": "[c:1]-[#7,#8,#16:2]"
        },
        "type": "two",
        "A": "Boronate",
        "B": "C-N, C-O, or C-S"
    },
    "Buchwald-Hartwig Amination": {
        "smarts": {#"[#7:1].[#17,#35,#53]([c:2]) >> [#7:1]-[c:2]",
            "1": "[#7h:1]",
            "2": "[#17,#35,#53]([c:2])",
            "3": "[#7:1]-[c:2]"
        },
        "type": "two",
        "A": "Amine",
        "B": "Aryl Halide"
    },
    "Buchwald-Hartwig C-O": {
            "smarts": {#"[#8:1].[#17,#35,#53]([c:2]) >> [#8:1]-[c:2]",
                "1": "[#8:1]",
                "2": "[#17,#35,#53]([c:2])",
                "3": "[#8:1]-[c:2]"
            },
            "type": "two",
            "A": "Phenol/Alcohol",
            "B": "Aryl Halide"
        },
    "Reductive Amination": {
        "smarts": {
            "1": "[C;X3;H0,H1:1](=[O])([#6])",
            "2": "[N;H1,H2:2]",
            "3": "[C:1]-[N:2]"
        },
        "type": "two",
        "A": "aldehyde or ketone",
        "B": "amine"
    },
    "Sulfonamide formation": {
        "smarts":{
            "1": "[S:1](=O)(=O)[Cl]",
            "2": "[#7h:2]",
            "3": "[S:1](=O)(=O)-[#7:2]"
        },
        "type": "two",
        "A": "sulfonyl chloride",
        "B": "amine"
    },
    "CuAAC \"click\" cycloaddition": {
        "smarts": {
            "1": "[#6:1]-[N:2]=[N+:3]=[N-:4]",
            "2": "[#6:5]#[C;H1:6]",
            "3": "[#6:1]-[n:2]1[n:3]=[n:4][c:6](-[#6:5])[cH:7]1"
        },
        "type": "two",
        "A": "azide",
        "B": "terminal alkyne"
    },
    "SE: Nitrogen deletion": {
        "smarts": "[#6X4&!a:1][NH&!a:2][#6X4&!a:3]>>[#6:1][#6:3]",
        "type": "one",
        "Starting Material": "Aliphatic C(sp3)-NH-C(sp3)",
        "Product": "Aliphatic C-C"
    },
    "SE: Isoxazole -> Pyrrole Swap": {
        "smarts": "[o:1]1[n:2][c:3][c:4][c:5]1>>[c:1]1=[n:2][c:3][c:4][c:5]1",
        "type": "one",
        "Starting Material": "Isoxazole ring",
        "Product": "Pyrrole ring"
    },
    "SE: Aryl C-to-N replacement": {
        "smarts": "[c;r6:1]([NH2:2])>>[n;r6:1]",
        "type": "one",
        "Starting Material": "Aniline, NH2",
        "Product": "Pyridine"
    },
    "Alcohol -> Carbonyl oxidation": {
        "smarts": "[#6h:1][OH:2]>>[#6:1]=[O:2]",
        "type": "one",
        "Starting Material": "C-OH",
        "Product": "C=O"
    },
    "Aldehyde -> Carboxylic acid oxidation":{
        "smarts": "[CX3H1:1](=[O:2])[#6:3]>>[CX3:1](=[O:2])([OH])[#6:3]",
        "type": "one",
        "Starting Material": "HC=O",
        "Product": "COOH"
    },
    "Sulfide -> Sulfoxide oxidation":{
        "smarts": "[#6:1][S:2][#6:3]>>[#6:1][S:2](=[O:4])[#6:3]", 
        "type": "one",
        "Starting Material": "Sulfide",
        "Product": "Sulfoxide"
    },
    "Sulfide -> Sulfone oxidation":{
        "smarts": "[#6:1][S:2][#6:3]>>[#6:1][S:2](=[O:4])(=[O:5])[#6:3]",
        "type": "one",
        "Starting Material": "Sulfide",
        "Product": "Sulfone"
    },
    "Sulfoxide -> Sulfone oxidation":{
        "smarts": "[#6:1][S:2](=[O:3])[#6:4]>>[#6:1][S:2](=[O:3])(=[O:5])[#6:4]",
        "type": "one",
        "Starting Material": "Sulfoxide",
        "Product": "Sulfone"
    },
    "Ketone -> Alcohol reduction":{
        "smarts": "[#6:1][CX3:2](=[O:3])[#6:4]>>[#6:1][CX4H1:2]([OH:3])[#6:4]",
        "type": "one",
        "Starting Material": "C=O",
        "Product": "C-OH"
    },
    "Nitro -> Amine reduction":{
        "smarts": "[#6:1][N+:2](=[O:3])[O-:4]>>[#6:1][NX3H2+0:2]",
        "type": "one",
        "Starting Material": "Nitro",
        "Product": "Amine"
    },
    "Nitrile -> Amine reduction":{
        "smarts": "[#6:1][C:2]#[N:3]>>[#6:1][CH2:2][NX3H2:3]",
        "type": "one",
        "Starting Material": "Nitrile",
        "Product": "Amine"
    },
    "Alkene -> Alkane reduction":{
        "smarts": "[#6:1]=[#6:2]>>[#6:1][#6:2]",
        "type": "one",
        "Starting Material": "C=C",
        "Product": "C-C"
    },
    "N-Alkylation": {
        "smarts": {#"[#7:1].[#17,#35,#53]([C:2]) >> [#7:1]-[C:2]",
            "1": "[#7;H1,H2:1]",
            "2": "[#17,#35,#53]([C:2])",
            "3": "[#7:1]-[C:2]"
        },
        "type": "two",
        "A": "Amine",
        "B": "Alkyl Halide"
    },
    "Ni/Photoredox Decarboxylative Cross-Coupling": {
        "smarts": {#"[#6:1]-C(=O)(O).Cl([#6:2])>>[#6:1]-[#6:2]",
            "1": "[#6:1]-C(=O)(O)",
            "2": "Cl([#6:2])",
            "3": "[#6:1]-[#6:2]"
        },
        "type": "two",
        "A": "-COOH",
        "B": "Aryl Cl"
    },
    "Fluorination": {
        "smarts": "[#6;H1,H2,H3:1] >> [#6:1]-F",
        "type": "one",
        "Starting Material": "C-H",
        "Product": "C-F"
    },
    "Fragment Coupling": {
        "smarts": {#"[*:1]-[#85].[#85]-[*:2]>>[*:1]-[*:2]",
            "1": "[*:1]-[#85]",
            "2": "[#85]-[*:2]",
            "3": "[*:1]-[*:2]"
        },
        "type": "two",
        "A": "-At",
        "B": "-At"
    },
    "Fluorination of Sulfonyl Chlorides": {
        "smarts": "[#6:1]S(=O)(=O)[Cl:2] >> [#6:1]S(=O)(=O)[F:2]",
        "type": "one",
        "Starting Material": "Sulfonyl Chloride",
        "Product": "Sulfonyl Fluoride"
    },
    "Sulfonyl Fluorides from Aryl-X": {
        "smarts": "[c:1][#35,#53] >> [c:1][S](=[O])(=[O])[F]",
        "type": "one",
        "Starting Material": "Aryl Bromide or Iodide",
        "Product": "Aryl Sulfonyl Fluoride"
    },
    "C-H to C-At Swap": {
        "smarts": "[#6;H1,H2,H3:1] >> [#6:1][#85]",
        "type": "one",
        "Starting Material": "C-H",
        "Product": "C-At"
    },
    "N-H to N-At Swap": {
        "smarts": "[#7;H1,H2:1] >> [#7:1][#85]",
        "type": "one",
        "Starting Material": "N-H",
        "Product": "N-At"
    },
    "O-H to O-At Swap": {
        "smarts": "[#8;H1:1] >> [#8:1][#85]",
        "type": "one",
        "Starting Material": "O-H",
        "Product": "O-At"
        },
    "At to H Swap": {
        "smarts": "[*:1][#85] >> [*:1]",
        "type": "one",
        "Starting Material": "any atom-At",
        "Product": "any atom-H"
    },
    "Methylation": {
        "smarts": "[#6,#7,#8;H1,H2,H3:1] >> [*:1][C]",
        "type": "one",
        "Starting Material": "C-H",
        "Product": "C-CH3"
    },
    "TriFluoroMethylation": {
        "smarts": "[*h:1] >> [*:1][C](F)(F)F",
        "type": "one",
        "Starting Material": "C-H",
        "Product": "C-CF3"
    },
    "Alkane -> Alkene": {
        "smarts": "[#6;H1,H2,H3:1][#6;H1,H2,H3:2] >> [#6:1]=[#6:2]",
        "type": "one",
        "Starting Material": "Ch-Ch",
        "Product": "C=C"
    },
    "Thioetherification": {
        "smarts": {
            "1": "[#16:1]-[#1:2]",
            "2": "[Cl,Br,I](-[#6:3])",
            "3": "[#6:3]-[#16:1]"
        },
        "type": "two",
        "A": "Thiol",
        "B": "Alkyl/Aryl Halide"
    },
    "Williamson Ether Synthesis": {
        "smarts": {
            "1": "[#8;H1:1]",
            "2": "[Cl,Br,I](-[#6;!c:3])", 
            "3": "[#8:1]-[#6;!c:3]"
        },
        "type": "two",
        "A": "Alcohol",
        "B": "Alkyl Halide"
    },
    "Aza Swap": {
        "smarts": "[Ch:1] >> [N:1]",
        "type": "one",
        "Starting Material": "C w/1+ Hs",
        "Product": "N"
    },
    "N-Aromatic Swap": {
        "smarts": "[ch:1] >> [n:1]",
        "type": "one",
        "Starting Material": "Arom. C",
        "Product": "Arom. N"
    },
    "Shrink Aliph. Ring": {
        "smarts": "[R:1]-[Ch2;r4,r5,r6,r7]-[R:2] >> [R:1]-[R:2]",
        "type": "one",
        "Starting Material": "Aliph. Ring C",
        "Product": "Atom Removed"
    },
    "Extend Aliph. Ring": {
        "smarts": "[A;r3,r4,r5,r6:1]-[A;r3,r4,r5,r6:2] >> [R:1]-[Ch2;R]-[R:2]",
        "type": "one",
        "Starting Material": "Aliphatic ring",
        "Product": "Ring 1 CH2 larger"
    },
    "Extend Alkyl Chain": {
        "smarts": "[C:1]-[*:2] >> [C:1]-[Ch2]-[*:2]",
        "type": "one",
        "Starting Material": "-C-any-",
        "Product": "-C-C-any-"
    },
    "Shrink Alkyl Chain": {
        "smarts": "[C:1]-[Ch2]-[*:2] >> [C:1]-[*:2]",
        "type": "one",
        "Starting Material": "-C-C-any-",
        "Product": "-C-any-"
    },
    "Cyclopropylation": {
        "smarts": "[Ch:1]-[Ch:2] >> [C:1]1-[Ch2]-[C:2]1",
        "type": "one",
        "Starting Material": "-C-C-",
        "Product": "-(cPr)-"
    },
    "Boc Deprotection": {
        "smarts": "[#7:1]C(=O)OC(C)(C)C >> [#7:1][H]",
        "type": "one",
        "Starting Material": "N-boc",
        "Product": "N-H"
    },
    "Ester Hydrolysis": {
        "smarts": "[C:1](=[O:2])[O:3][#6:4] >> [C:1](=[O:2])[OH:3]",
        "type": "one",
        "Starting Material": "Ester",
        "Product": "Carboxylic Acid"
    }
}

# --------------- Substructure Dictionary ---------------------
substructure_dictionary = {
    "Contains an Aromatic Ring":  "[a;r]",
    "Contains Aliphatic Ring":    "[C;R;!a]",
    "Contains Terminal Alkyne":   "[C;H1]#[C]",
    "Contains Terminal Alkene":   "[C;H2]=[C]",
    "Contains Internal Alkyne":   "[C;H0]#[C;H0]",
    "Contains Internal Alkene":   "[C;H0,H1]=[C;H0,H1]",
    "Contains Fluorine":          "[#6][#9]",
    "Contains Chlorine":          "[#6][#17]",
    "Contains Bromine":           "[#6][#35]",
    "Contains Iodine":            "[#6][#53]",
    "Contains Primary Amine":     "[NX3;H2;!$(NC=O)]",
    "Contains Secondary Amine":   "[NX3;H1;!$(NC=O)]",
    "Contains Tertiary Amine":    "[NX3;H0;!$(NC=O)]([#6])([#6])[#6]",
    "Contains Amide":             "[NX3][CX3](=O)",
    "Contains Nitro Group":       "[$([NX3](=O)=O),$([NX3+](=O)[O-])]",
    "Contains Nitrile":           "[CX2]#[NX1]",
    "Contains Isocyanate":        "[NX2]=[CX2]=O",
    "Contains Boronate":          "[BX3]([OX2])[OX2]",
    "Contains Phosphate":         "[PX4](=O)([OX2])([OX2])[OX2]",
    "Contains Thiol":             "[SX2H1]",
    "Contains Alcohol":           "[OX2H1]",
    "Contains Aldehyde":          "[CX3H1](=O)",
    "Contains Ketone":            "[CX3](=O)([#6])[#6]",
    "Contains Carboxylic Acid":   "[CX3](=O)[OX2H1]",
    "Contains Sulfonate":         "[SX4](=O)(=O)[OX2]",
    "Contains Sulfinate":         "[SX3](=O)[OX2]",
    "Contains Sulfoxide":         "[SX3](=O)([#6])[#6]",
    "Contains Sulfone":           "[SX4](=O)(=O)([#6])[#6]",
    "Contains Sulfonyl Chloride": "[SX4](=O)(=O)Cl",
    "Contains Sulfonyl Fluoride": "[SX4](=O)(=O)F",
    "Contains Vinyl Sulfone":     "[CX3H2]=[CX3H][SX4](=O)(=O)",
    "Contains Acrylamide":        "[CX3H2]=[CX3H][CX3](=O)[NX3]",
    "Contains Epoxide":           "[O;r3]1[C;r3][C;r3]1",
    "Contains Aziridine":         "[N;r3]1[C;r3][C;r3]1",
    "Contains Cyclopropane":      "C1CC1",
    "Contains Cyclopentane":      "C1CCCC1",
    "Contains Cyclohexane":       "C1CCCCC1",
    "Contains Cycloheptane":      "C1CCCCCC1",
    "Contains Piperidine":        "C1CCNCC1",
    "Contains Piperazine":        "C1CNCCN1",
    "Contains Pyrrolidine":       "C1CCNC1",
    "Contains Morpholine":        "C1COCCN1",
    "Contains Pyridine Ring":     "n1ccccc1",
    "Contains Pyrimidine Ring":   "c1cncnc1",
    "Contains Pyrrole Ring":      "[nH]1cccc1",
    "Contains Imidazole Ring":    "c1cnc[nH]1",
    "Contains Thiazole Ring":     "c1ncsc1",
    "Contains Ester":             "[CX3](=O)[OX2][#6]",
    "Contains Ether":             "[#6][OX2][#6]",
    "Contains Phenol":            "[c][OX2H1]",
    "Contains Aniline":           "[c][NX3;H1,H2]",
    "Contains Urea":              "[NX3][CX3](=O)[NX3]",
    "Contains Carbamate":         "[NX3][CX3](=O)[OX2]",
    "Contains Sulfonamide":       "[SX4](=O)(=O)[NX3]",
    "Contains Azide":             "[$([NX1-]=[NX2+]=[NX1]),$([NX1]#[NX2+][NX1-])]",
    "Contains Aryl Halide":       "[c][F,Cl,Br,I]",
    "Contains Alkyl Halide":      "[C;!a][Cl,Br,I]",
    "Contains Benzene Ring":      "c1ccccc1",
    "Contains Trifluoromethyl":   "[CX4](F)(F)F",
    "Contains Thiophene Ring":    "c1ccsc1",
    "Contains Furan Ring":        "c1ccoc1",
}

# --------------- Descriptor Filters Dictionary ---------------------
filters_dictionary = {
    "Lipinski Rule of Five": {
        "Info": "MW ≤ 500, logP ≤ 5, HBD ≤ 5, HBA ≤ 10.",
        "MW": {
            "max": 500.00
        },
        "logP": {
            "max": 5.00
        },
        "HBD": {
            "max": 5
        },
        "HBA": {
            "max": 10
        }
    },
    "Ghose Filter": {
        "Info": "160 ≤ MW ≤ 480, -0.4 ≤ logP ≤ 5.6, 40 ≤ atoms ≤ 70.",
        "MW": {
            "min": 160.00,
            "max": 480.00
        },
        "logP": {
            "min": -0.40,
            "max": 5.60
        },
        "AtomCount": {
            "min": 40,
            "max": 70
        }
    },
    "Veber Filter": {
        "Info": "RotB ≤ 10, TPSA ≤ 140 Å² (oral bioavailability).",
        "RotB": {
            "max": 10
        },
        "TPSA": {
            "max": 140.00
        }
    },
    "Egan Filter": {
        "Info": "logP ≤ 5.88 and TPSA ≤ 131 Å².",
        "logP": {
            "max": 5.88
        },
        "TPSA": {
            "max": 131.00
        }
    },
    "Muegge Filter": {
        "Info": "200 ≤ MW ≤ 600, -2 ≤ logP ≤ 5, RingCount ≤ 7, HBD ≤ 5, HBA ≤ 10, TPSA ≤ 150 Å².",
        "MW": {
            "min": 200.00,
            "max": 600.00
        },
        "logP": {
            "min": -2.00,
            "max": 5.00
        },
        "RingCount": {
            "max": 7
        },
        "HBD": {
            "max": 5
        },
        "HBA": {
            "max": 10
        },
        "TPSA": {
            "max": 150.00
        }
    },
    "Rule of Three": {
        "Info": "Fragment-like: MW ≤ 300, logP ≤ 3, HBD/HBA ≤ 3.",
        "MW": {
            "max": 300.00
        },
        "logP": {
            "max": 3.00
        },
        "HBD": {
            "max": 3
        },
        "HBA": {
            "max": 3
        }
    },
    "Oprea Lead-Like": {
        "Info": "MW 200-450, logP -1 to 4.5, HBD ≤5, HBA ≤8, RotB ≤8, aromatic rings ≤4.",
        "MW": {"min": 200.00, "max": 450.00},
        "logP": {"min": -1.00, "max": 4.50},
        "HBD": {"max": 5},
        "HBA": {"max": 8},
        "RotB": {"max": 8},
        "AromaticRings": {"max": 4},
    },
    "Oprea Drug-Like": {
        "Info": "HBD ≤1, 3 ≤ HBA ≤9, 3 ≤ RotB ≤7, 2 ≤ rings ≤3 (Oprea, 2000).",
        "HBD": {"max": 1},
        "HBA": {"min": 3, "max": 9},
        "RotB": {"min": 3, "max": 7},
        "RingCount": {"min": 2, "max": 3},
    },
    "Rule of Four": {
        "Info": "MW ≥400, logP ≥4, rings ≥4, HBA ≥4 (PPI-inhibitor-focused - deliberately opposite of drug-like).",
        "MW": {"min": 400.00},
        "logP": {"min": 4.00},
        "RingCount": {"min": 4},
        "HBA": {"min": 4},
    },
    "Mozziconacci Filter": {
        "Info": "Halogens ≤7, N ≥1, O ≥1, rings ≤6, RotB ≤15.",
        "HalogenCount": {"max": 7},
        "NitrogenCount": {"min": 1},
        "OxygenCount": {"min": 1},
        "RingCount": {"max": 6},
        "RotB": {"max": 15},
    },
    "Palm Filter": {
        "Info": "TPSA < 140 Å² (oral bioavailability, Palm et al. 1997).",
        "TPSA": {"max": 139.99},
    },
    "Murcko CNS Filter": {
        "Info": "MW 200-400, logP ≤5.2, HBA ≤4, HBD ≤3, RotB ≤7 (CNS-focused, Ajay/Bemis/Murcko 1999).",
        "MW": {"min": 200.00, "max": 400.00},
        "logP": {"max": 5.20},
        "HBA": {"max": 4},
        "HBD": {"max": 3},
        "RotB": {"max": 7},
    },
    "REOS Filter": {
        "Info": "MW 200-500, logP -5 to 5, HBD ≤5, HBA ≤10, RotB ≤8, atoms 15-50, charge -2 to 2.",
        "MW": {"min": 200.00, "max": 500.00},
        "logP": {"min": -5.00, "max": 5.00},
        "HBD": {"max": 5},
        "HBA": {"max": 10},
        "RotB": {"max": 8},
        "AtomCount": {"min": 15, "max": 50},
        "FormalCharge": {"min": -2, "max": 2},
    }
}

twoReactions = []
oneReactions = []
for x_, y_ in reactions_dictionary.items():   
    t_ = reactions_dictionary[x_]["type"]
    if t_ == "two" or t_ == "multi":
        twoReactions.append(x_)
    elif t_ == "one":
        oneReactions.append(x_)
    else:
        print("Error identifying reaction type for reaction: "+x_)
twoReactions.sort()
twoReactions.insert(0, "Multi Component")   
oneReactions.sort()
oneReactions.insert(0, "One Component")     

reaction = ""              
startup_log = ""
if SASCORE_SOURCE.startswith("rdkit-contrib"):
    startup_log += (
        f"\n[SAScore source: {SASCORE_SOURCE} - "
        f"Ertl-Schuffenhauer synthetic accessibility score]\n"
    )
else:
    startup_log += (
        "\n[SAScore unavailable: the RDKit/Ertl-Schuffenhauer "
        "SA_Score implementation could not be loaded.]\n"
    )
if not MATPLOTLIB_AVAILABLE:
    startup_log += f"\n[Chemical-space plotting disabled: {MATPLOTLIB_IMPORT_ERROR}]\n"

baseFiles = []         
reactantFiles = []     
baseSMILES = []        
initialBaseSMILES = [] 
reactantSMILES = []    
lastReactantSMILES = []
reactant_pool = []     
abbaVar = ""           
smarts = ""            
smis_a = []            
smis_b = []            
rxn_products = []      
exportPath = ""        
workingDirectory = os.path.expanduser("~")   
name = ""           
baseNum = 0         
frag_smarts = ""    
duplicated_smis = []
mol_cache = {}      
reaction_cache = {} 

def _get_cached_mol(smiles):
    if smiles not in mol_cache:
        mol_cache[smiles] = Chem.MolFromSmiles(smiles)
    return mol_cache[smiles]

def _get_cached_reaction(smarts_string):
    if smarts_string not in reaction_cache:
        reaction_cache[smarts_string] = AllChem.ReactionFromSmarts(smarts_string)
    return reaction_cache[smarts_string]

def roundDown(num, digits):
    scale = 10 ** abs(digits)
    if digits >= 0:
        return (math.floor(num * scale)) / scale
    return math.floor(num / scale) * scale

def roundUp(num, digits):
    scale = 10 ** abs(digits)
    if digits >= 0:
        return (math.ceil(num * scale)) / scale
    return math.ceil(num / scale) * scale

# --- Two-component reaction SMARTS assembly ---------------------------------
def genSMARTS(rxn):
    global reactions_dictionary
    rxn_smarts = reactions_dictionary[rxn]["smarts"]["1"]+"."+reactions_dictionary[rxn]["smarts"]["2"]+">>"+reactions_dictionary[rxn]["smarts"]["3"]
    return rxn_smarts

def invSMARTS(rxn):
    global reactions_dictionary
    rxn_smarts = reactions_dictionary[rxn]["smarts"]["2"]+"."+reactions_dictionary[rxn]["smarts"]["1"]+">>"+reactions_dictionary[rxn]["smarts"]["3"]
    return rxn_smarts

def intraSMARTS(rxn):
    global reactions_dictionary
    rxn_smarts = "("+reactions_dictionary[rxn]["smarts"]["1"]+"."+reactions_dictionary[rxn]["smarts"]["2"]+")>>"+reactions_dictionary[rxn]["smarts"]["3"]
    return rxn_smarts

# -----------------------------------------------------------------------
# Parallel molecule-processing functions
# -----------------------------------------------------------------------
def MultiPoolReaction(num, reactant_pool_list=None, smarts_string=None):
    global smarts
    if reactant_pool_list is None:
        reactant_pool_list = reactant_pool   
    if smarts_string is None:
        smarts_string = smarts               
    smile1 = reactant_pool_list[num]
    resulting_smile_list = []
    seen_smiles = set()
    mol1 = _get_cached_mol(smile1)
    rxn3 = _get_cached_reaction(smarts_string)
    for smile2 in reactant_pool_list[(num+1):]:  
        mol2 = _get_cached_mol(smile2)
        try:
            products = rxn3.RunReactants([mol1, mol2])
        except Exception:
            continue
        if not products:
            continue
        for i in range(len(products)):
            try:
                x = products[i][0]   
            except (IndexError, TypeError):
                continue
            if x is None:
                continue
            try:
                resulting_smile = Chem.MolToSmiles(x)
            except Exception:
                continue
            if resulting_smile in seen_smiles:
                continue
            seen_smiles.add(resulting_smile)
            resulting_smile_list.append(resulting_smile)
    return resulting_smile_list

def TwoComponentReaction(smile1, smis_b_list=None, smarts_string=None):
    global smis_b
    global smarts
    if smis_b_list is None:
        smis_b_list = smis_b
    if smarts_string is None:
        smarts_string = smarts
    resulting_smile_list = []
    seen_smiles = set()
    mol1 = _get_cached_mol(smile1)
    rxn3 = _get_cached_reaction(smarts_string)
    for smile2 in smis_b_list:
        mol2 = _get_cached_mol(smile2)
        try:
            products = rxn3.RunReactants([mol1, mol2])
        except Exception:
            continue
        if not products:
            continue
        for i in range(len(products)):
            try:
                x = products[i][0]
            except (IndexError, TypeError):
                continue
            if x is None:
                continue
            try:
                resulting_smile = Chem.MolToSmiles(x)
            except Exception:
                continue
            if resulting_smile in seen_smiles:
                continue
            seen_smiles.add(resulting_smile)
            resulting_smile_list.append(resulting_smile)
    return resulting_smile_list

def IntraMolecularReaction(smile1, intra_smarts_string=None):
    global intra_smarts
    if intra_smarts_string is None:
        intra_smarts_string = intra_smarts
    resulting_smile_list = []
    seen_smiles = set()
    mol1 = _get_cached_mol(smile1)
    rxn3 = _get_cached_reaction(intra_smarts_string)
    try:
        products = rxn3.RunReactants([mol1])
    except Exception:
        return resulting_smile_list
    if not products:
        return resulting_smile_list
    for i in range(len(products)):
        try:
            x = products[i][0]
        except (IndexError, TypeError):
            continue
        if x is None:
            continue
        try:
            resulting_smile = Chem.MolToSmiles(x)
        except Exception:
            continue
        if resulting_smile in seen_smiles:
            continue
        seen_smiles.add(resulting_smile)
        resulting_smile_list.append(resulting_smile)
    return resulting_smile_list

def OneComponentReaction(smile1, smarts_string=None):
    global smarts
    if smarts_string is None:
        smarts_string = smarts
    resulting_smile_list = []
    seen_smiles = set()
    mol1 = _get_cached_mol(smile1)
    rxn3 = _get_cached_reaction(smarts_string)
    try:
        products = rxn3.RunReactants([mol1])
    except Exception:
        return resulting_smile_list
    if not products:
        return resulting_smile_list
    for i in range(len(products)):
        try:
            x = products[i][0]
        except (IndexError, TypeError):
            continue
        if x is None:
            continue
        try:
            resulting_smile = Chem.MolToSmiles(x)
        except Exception:
            continue
        if resulting_smile in seen_smiles:
            continue
        seen_smiles.add(resulting_smile)
        resulting_smile_list.append(resulting_smile)
    return resulting_smile_list

def OneComponentReactionBatch(smiles_batch, smarts_string):
    resulting_smiles = []
    seen_smiles = set()

    rxn = _get_cached_reaction(smarts_string)

    if rxn is None or rxn.GetNumReactantTemplates() != 1:
        return resulting_smiles

    reactant_template = rxn.GetReactantTemplate(0)

    for smile in smiles_batch:
        mol = Chem.MolFromSmiles(smile)

        if mol is None:
            continue
        if not mol.HasSubstructMatch(reactant_template):
            continue

        try:
            products = rxn.RunReactants((mol,))
        except Exception:
            continue

        if not products:
            continue

        for product_set in products:
            if not product_set:
                continue

            product = product_set[0]

            if product is None:
                continue

            try:
                product_smiles = Chem.MolToSmiles(product)
            except Exception:
                continue

            if product_smiles in seen_smiles:
                continue

            seen_smiles.add(product_smiles)
            resulting_smiles.append(product_smiles)

    return resulting_smiles

def AssignIsomers(smile1):
    resulting_smile_list = []
    mol_ = _get_cached_mol(smile1)

    if mol_ is None:
        return resulting_smile_list
    
    try:
        for chiral_mol in EnumerateStereoisomers(mol_):
            try:
                resulting_smile_list.append(Chem.MolToSmiles(chiral_mol))
            except Exception:
                continue
    except Exception:
        pass

    return resulting_smile_list

def AssignIsomersBatch(smiles_batch):
    results = []

    for smile in smiles_batch:
        results.extend(AssignIsomers(smile))

    return results

def removeIfMatch(smi, frag_smarts_string=None):
    global frag_smarts

    if frag_smarts_string is None:
        frag_smarts_string = frag_smarts

    resulting_smile_list = []
    mol = _get_cached_mol(smi)

    if mol is not None and mol.HasSubstructMatch(Chem.MolFromSmarts(frag_smarts_string)) == False:
        resulting_smile_list.append(smi)

    return resulting_smile_list

def format_filter_ranges(filter_ranges):
    parts = []

    for descriptor, values in filter_ranges.items():

        min_val = values["min"]
        max_val = values["max"]
        original_min = values["original_min"]
        original_max = values["original_max"]

        if min_val != original_min or max_val != original_max:
            parts.append(f"{descriptor}={min_val:g}-{max_val:g}")

    return "; ".join(parts)

def DeDuplicate(smis_list):
    return list(set(smis_list))

def _chunk_list(items, chunk_size):
    for i in range(0, len(items), chunk_size):
        yield items[i:i + chunk_size]

def _collect_pool_results(executor, fn, items, *args):
    if not items:
        return []
    
    pending = [executor.submit(fn, item, *args) for item in items]
    collected = []

    for future in concurrent.futures.as_completed(pending):
        collected.extend(future.result())

    return collected


def _drain_with_pool(fn, items, *args):
    if not items:
        return []
    
    with concurrent.futures.ProcessPoolExecutor() as pool:
        pending = [pool.submit(fn, item, *args) for item in items]
        collected = []

        for future in concurrent.futures.as_completed(pending):
            collected.extend(future.result())

        return collected


def Deprotection(smile1): #Placeholder
    return [smile1]


class UgiReaction(QThread):
    finished = pyqtSignal()
    def run(self):
        global frag_smarts
        global rxn_products
        global smarts
        global intra_smarts
        global reactant_pool

        rxn_products = []
        step_products = []
        imines = []       
        temp_products = []

        #Step 1: Imine formation  -> need to switch smarts to an easily invertable format
        smarts = "[C!$(C([#7,#8])=O):1]=[O].[Nh2!$(NC=O):2]>>[C:1]=[N:2]"
        intra_smarts = "([C!$(C([#7,#8])=O):1]=[O].[Nh2!$(NC=O):2])>>[C:1]=[N:2]" 

        with concurrent.futures.ProcessPoolExecutor() as executor:
            step_products.extend(_collect_pool_results(executor, MultiPoolReaction, range(0, len(reactant_pool)), reactant_pool, smarts))

        smarts = "[Nh2!$(NC=O):2].[C!$(C([#7,#8])=O):1]=[O]>>[C:1]=[N:2]"

        with concurrent.futures.ProcessPoolExecutor() as executor:
            step_products.extend(_collect_pool_results(executor, MultiPoolReaction, range(0, len(reactant_pool)), reactant_pool, smarts))
        with concurrent.futures.ProcessPoolExecutor() as executor:
            step_products.extend(_collect_pool_results(executor, IntraMolecularReaction, reactant_pool, intra_smarts))
        
        imines.extend(step_products)          
        reactant_pool.extend(step_products)   
        step_products = []

        #Step 2: Nucleophilic attack of isocyanide to imine
        smarts = "[#6:1][N+:2]#[C:3].[C:4]=[N:5]>>[#6:1][N+0:2][C-0:3](=O)[C:4][N-:5]"
        intra_smarts = "([#6:1][N+:2]#[C:3].[C:4]=[N:5])>>[#6:1][N+0:2][C-0:3](=O)[C:4][N-:5]"

        with concurrent.futures.ProcessPoolExecutor() as executor:
            step_products.extend(_collect_pool_results(executor, MultiPoolReaction, range(0, len(reactant_pool)), reactant_pool, smarts))

        smarts = "[C:4]=[N:5].[#6:1][N+:2]#[C:3]>>[#6:1][N+0:2][C-0:3](=O)[C:4][N-:5]"

        with concurrent.futures.ProcessPoolExecutor() as executor:
            step_products.extend(_collect_pool_results(executor, MultiPoolReaction, range(0, len(reactant_pool)), reactant_pool, smarts))
        with concurrent.futures.ProcessPoolExecutor() as executor:
            step_products.extend(_collect_pool_results(executor, IntraMolecularReaction, reactant_pool, intra_smarts))

        reactant_pool.extend(step_products)
        step_products = []

        reactant_pool_copy = []
        reactant_pool_copy.extend(reactant_pool)
        reactant_pool = []

        for mol in reactant_pool_copy:      
            if mol in imines: continue
            reactant_pool.append(mol)

        reactant_pool_copy = []

        #Step 3: Formation of final peptide by amide coupling carboxylic acid to N
        smarts = "[C:1](=[O:2])O.[N-:3]>>[C:1](=[O:2])[N-0:3]"
        intra_smarts = "([C:1](=[O:2])O.[N-:3])>>[C:1](=[O:2])[N-0:3]"

        with concurrent.futures.ProcessPoolExecutor() as executor:
            step_products.extend(_collect_pool_results(executor, MultiPoolReaction, range(0, len(reactant_pool)), reactant_pool, smarts))

        smarts = "[N-:3].[C:1](=[O:2])O>>[C:1](=[O:2])[N-0:3]"

        with concurrent.futures.ProcessPoolExecutor() as executor:
            step_products.extend(_collect_pool_results(executor, MultiPoolReaction, range(0, len(reactant_pool)), reactant_pool, smarts))
        with concurrent.futures.ProcessPoolExecutor() as executor:  
            step_products.extend(_collect_pool_results(executor, IntraMolecularReaction, reactant_pool, intra_smarts))

        frag_smarts = "[N-]"    
        temp_products = []

        with concurrent.futures.ProcessPoolExecutor() as executor:
            temp_products.extend(_collect_pool_results(executor, removeIfMatch, step_products, frag_smarts))

        step_products = []

        with concurrent.futures.ProcessPoolExecutor() as executor:
            rxn_products.extend(_collect_pool_results(executor, AssignIsomers, temp_products))
       
        temp_products = []
        rxn_products.extend(step_products)
        reactant_pool = []
        self.finished.emit() 

class RxnQThread(QThread):
    finished = pyqtSignal()

    def run(self):
        global smarts
        global intra_smarts
        global smis_a
        global smis_b
        global rxn_products
        global baseSMILES
        global reactantSMILES
        global start
        global finish
        global reactions_dictionary
        global reaction
        global abbaVar
        global reactant_pool
        temp_products = []

        if reactions_dictionary[reaction]["type"] == "two":
            if abbaVar == "Reactant Pool":
                smarts = genSMARTS(reaction)

                with concurrent.futures.ProcessPoolExecutor() as executor:
                    rxn_products.extend(_collect_pool_results(executor, MultiPoolReaction, range(0, len(reactant_pool)-1), reactant_pool, smarts))
                smarts = invSMARTS(reaction)

                with concurrent.futures.ProcessPoolExecutor() as executor:
                    rxn_products.extend(_collect_pool_results(executor, MultiPoolReaction, range(0, len(reactant_pool)-1), reactant_pool, smarts))
                intra_smarts = intraSMARTS(reaction)

                with concurrent.futures.ProcessPoolExecutor() as executor:
                    rxn_products.extend(_collect_pool_results(executor, IntraMolecularReaction, reactant_pool, intra_smarts))
            else:
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    rxn_products.extend(_collect_pool_results(executor, TwoComponentReaction, smis_a, smis_b, smarts))
        elif reactions_dictionary[reaction]["type"] == "one":
            smarts = reactions_dictionary[reaction]["smarts"]
            retained_starting_materials = list(rxn_products)
            batch_size = 2000
            batches = _chunk_list(smis_a, batch_size)

            generated_products = []

            with concurrent.futures.ProcessPoolExecutor() as executor:
                futures = [
                    executor.submit(
                        OneComponentReactionBatch,
                        batch,
                        smarts
                    )
                    for batch in batches
                ]

                for future in concurrent.futures.as_completed(futures):
                    generated_products.extend(future.result())

            generated_products = DeDuplicate(generated_products)
            enumerated_products = []

            if generated_products:
                isomer_batches = _chunk_list(generated_products, batch_size)

                with concurrent.futures.ProcessPoolExecutor() as executor:
                    futures = [executor.submit(AssignIsomersBatch, batch) for batch in isomer_batches]

                    for future in concurrent.futures.as_completed(futures):
                        enumerated_products.extend(future.result())

            rxn_products = retained_starting_materials + enumerated_products
        else: print("Error, invalid reaction type")
        self.finished.emit()

class imageViewer(QWidget):
    global viewerList
    def __init__(self):
        super().__init__()
        global viewerList
        global baseNum
        layout = QVBoxLayout()
        self.createImageViewer()
        self.createViewerBtns()
        layout.addLayout(self.viewerBtns)
        layout.addLayout(self.imageViewerLayout)
        self.setLayout(layout)

    def createImageViewer(self):
        global viewerList
        global baseNum
        from rdkit.Chem import Draw 
        dopts = Draw.rdMolDraw2D.MolDrawOptions()
        dopts.clearBackground = False 
        dopts.addStereoAnnotation = True
        dopts.annotationFontScale = 1.25
        dopts.drawMolsSameScale = False
        if QPalette().base().color().lightness() <= 130: 
            Draw.rdMolDraw2D.SetDarkMode(dopts)   
        viewerMols = []
        self.imageViewerLayout = QGridLayout()
        imageViewerWidget = QSvgWidget()
        for z in viewerList[baseNum:baseNum+25]:   
            viewerMols.append(Chem.MolFromSmiles(z))
        svg_data = Draw.MolsToGridImage(viewerMols, molsPerRow=5, useSVG=True, drawOptions=dopts)
        svg_bytes = bytearray(svg_data, encoding='utf-8')
        imageViewerWidget.renderer().load(svg_bytes)
        self.imageViewerLayout.addWidget(imageViewerWidget)

    def createViewerBtns(self): 
        global viewerList
        global baseNum
        self.viewerBtns = QGridLayout()
        backBtn = QPushButton("Prev. 25")
        backBtn.clicked.connect(self.goBack)
        numLabel = QLabel(str(baseNum+1)+" - "+str(min((baseNum+25), len(viewerList)))+" of "+str(len(viewerList)))
        numLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        fwdBtn = QPushButton("Next 25")
        fwdBtn.clicked.connect(self.goFwd)

        if len(viewerList) <= baseNum+25:
            fwdBtn.setDisabled(True)     
        else:
            fwdBtn.setDisabled(False)
        if baseNum == 0:
            backBtn.setDisabled(True)
        elif baseNum != 0:
            backBtn.setDisabled(False)

        self.viewerBtns.addWidget(backBtn, 0, 0)
        self.viewerBtns.addWidget(numLabel, 0, 1)
        self.viewerBtns.addWidget(fwdBtn, 0, 2)

    def goFwd(self):
        global baseNum
        self.close()
        if baseNum+25 < len(viewerList):
            baseNum = baseNum+25        
        self.w = imageViewer()
        self.w.show()

    def goBack(self):
        global baseNum
        self.close()
        baseNum = baseNum-25
        self.w = imageViewer()
        self.w.show()

class ExportWorker(QObject):
    progress = pyqtSignal(int) 
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)    

    def __init__(self, smiles_list, descriptor_lookup, export_path, file_name, file_type, include_properties=False):
        super().__init__()
        self.smiles_list = smiles_list
        self.descriptor_lookup = descriptor_lookup  
        self.export_path = export_path
        self.file_name = file_name
        self.file_type = file_type
        self.include_properties = include_properties

    def run(self):
        try:
            if self.file_type not in (
                ".sdf (Molecules only)",
                ".sdf (Molecules + Properties)",
                ".mol (Individual files)",
                "SMILES as .csv"
            ):
                self.error.emit(f"Unsupported export file type: {self.file_type}")
                return
            
            total = max(1, len(self.smiles_list))
            exported = 0
            skipped = 0
            skipped_smiles = []
            export_rows = []   

            writer = None

            if self.file_type in (".sdf (Molecules only)", ".sdf (Molecules + Properties)"):
                writer = Chem.SDWriter(os.path.join(self.export_path, self.file_name + '.sdf'))

            for i, p in enumerate(self.smiles_list):
                try:
                    x = Chem.MolFromSmiles(p)
                    if x is None:
                        raise ValueError("could not parse SMILES")

                    if self.include_properties:
                        props = {"SMILES": p}

                        if p in self.descriptor_lookup.index:
                            row = self.descriptor_lookup.loc[p]
                            for col in self.descriptor_lookup.columns:
                                props[col] = row[col]
                        else:
                            props.update(_compute_props(p))

                        props = _normalize_descriptor_props(props)
                        _set_mol_properties(x, props)

                    if self.file_type in (
                        ".sdf (Molecules only)",
                        ".sdf (Molecules + Properties)"
                    ):
                        writer.write(x)

                    elif self.file_type == ".mol (Individual files)":
                        Chem.MolToMolFile(
                            x,
                            os.path.join(
                                self.export_path,
                                self.file_name + "_" + str(exported + 1) + '.mol'
                            )
                        )
                    elif self.file_type == "SMILES as .csv":
                        csv_row = {"SMILES": p}
                        for col in self.descriptor_lookup.columns:
                            if col == "CanonSMILES":
                                continue 
                            csv_row[col] = props.get(col)
                        if "TanimotoSimilarity" in props:
                            csv_row["TanimotoSimilarity"] = props.get("TanimotoSimilarity")
                            csv_row["SimilarityPercent"] = props.get("SimilarityPercent")
                            csv_row["SimilarityDescription"] = props.get("SimilarityDescription")
                        export_rows.append(csv_row)
                    exported += 1
                except Exception as e:
                    skipped += 1
                    skipped_smiles.append(p)

                if i % max(1, total // 100) == 0:
                    self.progress.emit(int(100 * i / total))

            if writer is not None:
                writer.close()
            if self.file_type == "SMILES as .csv":
                export_df = pd.DataFrame(export_rows)
                export_df["SAScore_source"] = SASCORE_SOURCE
                export_df.to_csv(self.export_path + '/' + self.file_name + '_SMILES.csv', index=False)

            self.progress.emit(100)
            self.finished.emit({"exported": exported, "skipped": skipped, "skipped_smiles": skipped_smiles})
        except Exception as exc:
            self.error.emit(f"Export failed: {exc}")

class SplitWorker(QObject):
    progress = pyqtSignal(int)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, smiles_list, output_dir, base_name, batch_size=3500, descriptor_lookup=None):
        super().__init__()
        self.smiles_list = smiles_list
        self.output_dir = output_dir
        self.base_name = base_name
        self.batch_size = max(1, int(batch_size))
        self.descriptor_lookup = descriptor_lookup

    def run(self):
        try:
            if not self.smiles_list:
                self.error.emit("No library is loaded to split.")
                return
            if not os.path.isdir(self.output_dir):
                os.makedirs(self.output_dir, exist_ok=True)

            total = len(self.smiles_list)
            batch_num = 1
            mols_in_batch = []
            written = 0
            skipped = 0

            def flush_batch():
                nonlocal batch_num, mols_in_batch
                if not mols_in_batch:
                    return
                output_file = os.path.join(self.output_dir, f"{self.base_name}_batch_{batch_num}.sdf")
                with Chem.SDWriter(output_file) as writer:
                    for mol in mols_in_batch:
                        writer.write(mol)
                mols_in_batch = []
                batch_num += 1

            for idx, smi in enumerate(self.smiles_list):
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    skipped += 1
                    continue

                props = {"SMILES": smi}
                if self.descriptor_lookup is not None and smi in self.descriptor_lookup.index:
                    row = self.descriptor_lookup.loc[smi]
                    for col in self.descriptor_lookup.columns:
                        props[col] = row[col]
                else:
                    props.update(_compute_props(smi))
                _set_mol_properties(mol, _normalize_descriptor_props(props))

                mols_in_batch.append(mol)
                written += 1
                if len(mols_in_batch) == self.batch_size:
                    flush_batch()

                if idx % max(1, total // 100) == 0:
                    self.progress.emit(int(100 * idx / total))

            flush_batch()
            self.progress.emit(100)
            self.finished.emit({
                "file_count": batch_num - 1,
                "total_molecules": written,
                "skipped": skipped,
            })
        except Exception as exc:
            self.error.emit(f"Splitting failed: {exc}")

class DescriptorWorker(QObject):
    progress = pyqtSignal(int)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, smiles_list, batch_size=2000):
        super().__init__()
        self.smiles_list = smiles_list
        self.batch_size = batch_size

    def run(self):
        try:
            total = len(self.smiles_list)

            if total == 0:
                self.finished.emit(
                    pd.DataFrame(columns=["SMILES"] + DESCRIPTOR_COLUMNS)
                )
                return

            batches = list(
                _chunk_list(
                    self.smiles_list,
                    self.batch_size
                )
            )

            rows = []
            completed = 0

            with concurrent.futures.ProcessPoolExecutor() as executor:
                futures = [
                    executor.submit(
                        _compute_props_batch,
                        batch
                    )
                    for batch in batches
                ]

                for future in concurrent.futures.as_completed(futures):
                    batch_rows = future.result()
                    rows.extend(batch_rows)

                    completed += len(batch_rows)

                    progress = int(
                        completed / total * 100
                    )

                    self.progress.emit(
                        min(progress, 100)
                    )

            df = pd.DataFrame(rows)

            self.progress.emit(100)
            self.finished.emit(df)

        except Exception as exc:
            self.error.emit(
                f"Descriptor calculation failed: {exc}"
            )

class FilterWorker(QObject):
    progress = pyqtSignal(int) 
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)    

    def __init__(self, df, filter_ranges, filter_label="Custom"):
        super().__init__()
        self.df = df
        self.filter_ranges = filter_ranges
        self.filter_label = filter_label

    def run(self):
        try:
            if self.df is None or self.df.empty:
                self.finished.emit({"filters": [self.filter_label], "final_size": 0, "key_characteristics": {}})
                return

            df = self.df.copy()
            mask = pd.Series(True, index=df.index)

            for item, values in self.filter_ranges.items():
                min_value = values["min"]
                max_value = values["max"]

                original_min = values["original_min"]
                original_max = values["original_max"]

                if min_value <= original_min and max_value >= original_max:
                    continue
                
                mask &= df[item].between(
                    min_value,
                    max_value,
                    inclusive="both"
                )

            df = df.loc[mask].copy()

            self.progress.emit(90)
            final_size = len(df)
            kc = {} 
            if final_size:
                kc = {
                    "avg_MW": (round(float(df["MW"].mean()), 1) if "MW" in df.columns else None),
                    "avg_logP": (round(float(df["logP"].mean()), 2) if "logP" in df.columns else None),
                    "avg_SAScore": (round(float(df["SAScore"].mean()), 2) if "SAScore" in df.columns and not df["SAScore"].isna().all() else None),
                    "aromatic_rings≥2_%": (round(float((df["AromaticRings"] >= 2).mean() * 100), 1) if "AromaticRings" in df.columns else None),
                }

            self.progress.emit(100)
            
            filtered_df = df.copy()
            self.finished.emit({
                "filters": [self.filter_label],
                "final_size": final_size,
                "key_characteristics": kc,
                "filtered_df": filtered_df,
            })
        except Exception as exc:
            self.error.emit(f"Filtration failed: {exc}")

class SubstructureFilterWorker(QObject):
    progress = pyqtSignal(int)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, smiles_list, smarts_list, match_mode="all", exclude_pains=False,
                 exclude_brenk=False, exclude_nih=False, exclude_chembl=False):
        super().__init__()
        self.smiles_list = smiles_list
        self.smarts_list = smarts_list
        self.match_mode = match_mode
        self.exclude_pains = exclude_pains
        self.exclude_brenk = exclude_brenk
        self.exclude_nih = exclude_nih
        self.exclude_chembl = exclude_chembl

    def run(self):
        try:
            patterns = [Chem.MolFromSmarts(s) for s in self.smarts_list]
            if any(p is None for p in patterns):
                self.error.emit("One or more substructure patterns failed to parse.")
                return
            combine = all if self.match_mode == "all" else any
            catalogs = []
            if self.exclude_pains:
                catalogs.append(_get_pains_catalog())
            if self.exclude_brenk:
                catalogs.append(_get_brenk_catalog())
            if self.exclude_nih:
                catalogs.append(_get_nih_catalog())
            if self.exclude_chembl:
                catalogs.append(_get_chembl_catalog())

            total = max(1, len(self.smiles_list))
            filtered = []
            for idx, smi in enumerate(self.smiles_list):
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    passes_substructure = (not patterns) or combine(mol.HasSubstructMatch(p) for p in patterns)
                    passes_catalogs = all(not cat.HasMatch(mol) for cat in catalogs)
                    if passes_substructure and passes_catalogs:
                        filtered.append(smi)
                if idx % max(1, total // 100) == 0:
                    self.progress.emit(int(90 * idx / total))

            self.progress.emit(100)
            self.finished.emit({
                "filtered_smiles": DeDuplicate(filtered),
                "before": len(self.smiles_list),
                "after": len(filtered),
            })
        except Exception as exc:
            self.error.emit(f"Substructure filtering failed: {exc}")

class FilterByMoietyDialog(QDialog):
    def __init__(self, parent=None, checked_labels=None, match_mode="all", exclude_pains=False,
                 exclude_brenk=False, exclude_nih=False, exclude_chembl=False):
        super().__init__(parent)
        self.setWindowTitle("Filter Library by Substructure")
        self.resize(320, 360)

        self.selected_filters = []
        self.selected_labels = []
        self.match_mode = match_mode 

        checked_labels = checked_labels or set()

        self.list_widget = QListWidget()
        for label in sorted(substructure_dictionary):
            item = QListWidgetItem(label)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if label in checked_labels else Qt.CheckState.Unchecked)
            self.list_widget.addItem(item)

        select_all_row = QHBoxLayout()
        self.selectAllBtn = QPushButton("Select All")
        self.selectAllBtn.clicked.connect(lambda: self._set_all(Qt.CheckState.Checked))
        self.clearAllBtn = QPushButton("Clear All")
        self.clearAllBtn.clicked.connect(lambda: self._set_all(Qt.CheckState.Unchecked))
        select_all_row.addWidget(self.selectAllBtn)
        select_all_row.addWidget(self.clearAllBtn)

        self.matchAllRadio = QRadioButton("Match ALL checked fragments (AND)")
        self.matchAnyRadio = QRadioButton("Match ANY checked fragment (OR)")
        self.excludePainsCheckBox = QCheckBox("Also exclude PAINS-flagged compounds")
        self.excludePainsCheckBox.setChecked(exclude_pains)
        self.excludeBrenkCheckBox = QCheckBox("Also exclude Brenk-flagged compounds")
        self.excludeBrenkCheckBox.setChecked(exclude_brenk)
        self.excludeNihCheckBox = QCheckBox("Also exclude NIH-flagged compounds")
        self.excludeNihCheckBox.setChecked(exclude_nih)
        self.excludeChemblCheckBox = QCheckBox("Also exclude ChEMBL-curated alerts (Glaxo/Dundee/BMS/etc.)")
        self.excludeChemblCheckBox.setChecked(exclude_chembl)
        mode_group = QButtonGroup(self)
        mode_group.addButton(self.matchAllRadio)
        mode_group.addButton(self.matchAnyRadio)
        if match_mode == "any":
            self.matchAnyRadio.setChecked(True)
        else:
            self.matchAllRadio.setChecked(True)
        mode_row = QVBoxLayout()
        mode_row.addWidget(self.matchAllRadio)
        mode_row.addWidget(self.matchAnyRadio)

        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.on_save)
        button_box.rejected.connect(self.reject)

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Select fragments:"))
        layout.addLayout(select_all_row)
        layout.addWidget(self.list_widget)
        layout.addWidget(self.excludePainsCheckBox)
        layout.addWidget(self.excludeBrenkCheckBox)
        layout.addWidget(self.excludeNihCheckBox)
        layout.addWidget(self.excludeChemblCheckBox)
        layout.addLayout(mode_row)
        layout.addWidget(button_box)
        self.setLayout(layout)

    def _set_all(self, state):
        for i in range(self.list_widget.count()):
            self.list_widget.item(i).setCheckState(state)

    def on_save(self):
        checked_items = [
            self.list_widget.item(i)
            for i in range(self.list_widget.count())
            if self.list_widget.item(i).checkState() == Qt.CheckState.Checked
        ]
        self.selected_filters = [substructure_dictionary[item.text()] for item in checked_items]
        self.selected_labels = [item.text() for item in checked_items]
        self.match_mode = "any" if self.matchAnyRadio.isChecked() else "all"
        self.exclude_pains = self.excludePainsCheckBox.isChecked()
        self.exclude_brenk = self.excludeBrenkCheckBox.isChecked()
        self.exclude_nih = self.excludeNihCheckBox.isChecked()
        self.exclude_chembl = self.excludeChemblCheckBox.isChecked()
        self.accept()

class FiltrationWindow(QWidget):
    def __init__(self, parent=None, df=None):
        super().__init__(parent)
        global filters_dictionary
        self.df = df.copy()
        numeric_df = self.df.select_dtypes(include=['number'])
        self.filters = [col for col in numeric_df.columns if self.df[col].notna().any()]
        self.int_filters = [col for col in self.df.select_dtypes(include=['int']).columns if col in self.filters]
       
        self.original_ranges = {item: {"min": roundDown(self.df[item].min(), 2), "max": roundUp(self.df[item].max(), 2)} for item in self.filters}
        self.setWindowTitle("Library Filtration")
        self.resize(800, 420)
        self.selected_filters = []

        self.active_filter_description = ""
        self.active_filter_label = "Custom"

        filtrationWindowLayout = QGridLayout(self)

        existingFilterSelectionLayout = QGridLayout()
        self.preconfiguredFiltersGroupBox = QGroupBox()
        self.preconfiguredFiltersGroupBox.setLayout(existingFilterSelectionLayout)
        filtrationWindowLayout.addWidget(self.preconfiguredFiltersGroupBox, 0, 0)
        self.filterOptionsComboBox = QComboBox()
        self.filterDescription = QLineEdit("Description:")
        self.filterDescription.setReadOnly(True)
        self.filterOptionsComboBox.addItems(["Select"] + sorted(list(filters_dictionary.keys())))
        self.filterOptionsComboBox.currentTextChanged.connect(self.describeFilter)
        existingFilterSelectionLayout.addWidget(self.filterOptionsComboBox, 0, 0)
        existingFilterSelectionLayout.addWidget(self.filterDescription, 0, 1)

        self.applyPresetsBtn = QPushButton("Set Values")
        self.applyPresetsBtn.setFixedSize(120, 30)
        self.applyPresetsBtn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.applyPresetsBtn.clicked.connect(self.setFilterValues)
        existingFilterSelectionLayout.addWidget(self.applyPresetsBtn, 0, 2)
        self.applyPresetsBtn.setDisabled(True)

        self.resetFiltersBtn = QPushButton("Reset Filters")
        self.resetFiltersBtn.setFixedSize(120, 30)
        self.resetFiltersBtn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.resetFiltersBtn.clicked.connect(self.resetFilters)
        existingFilterSelectionLayout.addWidget(self.resetFiltersBtn, 0, 3)
        self.resetFiltersBtn.setEnabled(True)

        mainFiltrationLayout = QVBoxLayout()
        mainFiltrationLayout.setContentsMargins(8, 8, 8, 8)
        filtrationWindowLayout.addLayout(mainFiltrationLayout, 1, 0)

        self.FiltrationDialBox = QGroupBox("Custom Filtration Criteria:")
        spin_box_layout = QGridLayout(self.FiltrationDialBox)
        self.spin_box = {}
        row = 0
        spin_box_layout.addWidget(QLabel("Parameter", alignment=Qt.AlignmentFlag.AlignCenter), row, 0)
        spin_box_layout.addWidget(QLabel("Current Library Range", alignment=Qt.AlignmentFlag.AlignCenter), row, 1)
        spin_box_layout.addWidget(QLabel("Min", alignment=Qt.AlignmentFlag.AlignCenter), row, 2)
        spin_box_layout.addWidget(QLabel("Max", alignment=Qt.AlignmentFlag.AlignCenter), row, 3)
        row = row + 1
        for item in self.filters:  
            self.spin_box.update({item: {"label": QLabel(item), "min_spin_box": QDoubleSpinBox(), "max_spin_box": QDoubleSpinBox(), "min_value": 0, "max_value": 0, "range_label": QLabel(''), "min": self.original_ranges[item]["min"], "max": self.original_ranges[item]["max"]}})

            if item in self.int_filters:
                self.spin_box[item]["min_spin_box"] = QSpinBox()
                self.spin_box[item]["max_spin_box"] = QSpinBox()
                self.spin_box[item]["min"] = int(self.spin_box[item]["min"])
                self.spin_box[item]["max"] = int(self.spin_box[item]["max"])
            self.spin_box[item]["min_value"] = self.spin_box[item]["min"]
            self.spin_box[item]["max_value"] = self.spin_box[item]["max"]
            self.spin_box[item]["min_spin_box"].setRange(self.spin_box[item]["min"], self.spin_box[item]["max"])
            self.spin_box[item]["min_spin_box"].setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.spin_box[item]["min_spin_box"].setValue(self.spin_box[item]["min"])
            self.spin_box[item]["min_spin_box"].valueChanged.connect(self.updateLabels)
            self.spin_box[item]["max_spin_box"].setRange(self.spin_box[item]["min"], self.spin_box[item]["max"])
            self.spin_box[item]["max_spin_box"].setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.spin_box[item]["max_spin_box"].setValue(self.spin_box[item]["max"])
            self.spin_box[item]["max_spin_box"].valueChanged.connect(self.updateLabels)
            if item in self.int_filters:
                self.spin_box[item]["range_label"].setText(f"{int(self.spin_box[item]['min_value'])} - {int(self.spin_box[item]['max_value'])}")
            else:
                self.spin_box[item]["range_label"].setText(f"{self.spin_box[item]['min_value']:.2f} - {self.spin_box[item]['max_value']:.2f}")
            self.spin_box[item]["range_label"].setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.spin_box[item]["label"].setAlignment(Qt.AlignmentFlag.AlignCenter)
            spin_box_layout.addWidget(self.spin_box[item]["label"], row, 0)
            spin_box_layout.addWidget(self.spin_box[item]["range_label"], row, 1)
            spin_box_layout.addWidget(self.spin_box[item]["min_spin_box"], row, 2)
            spin_box_layout.addWidget(self.spin_box[item]["max_spin_box"], row, 3)
            row=row+1
        self._refresh_current_library_ranges()
        mainFiltrationLayout.addWidget(self.FiltrationDialBox)
        self.filterDfBtn = QPushButton("Apply Above Filters to Library")
        self.filterDfBtn.clicked.connect(self.start_filtration)
        mainFiltrationLayout.addWidget(self.filterDfBtn)
        self.viewFilteredLibraryBtn = QPushButton("View Filtered Library")
        self.viewFilteredLibraryBtn.clicked.connect(self.showFilteredLibraryTable)
        self.viewFilteredLibraryBtn.setDisabled(True)
        mainFiltrationLayout.addWidget(self.viewFilteredLibraryBtn)
        self.progressBox = QGroupBox("Filtration in progress")
        pb_layout = QVBoxLayout(self.progressBox)
        self.progressLabel = QLabel("Waiting to start…")
        self.progressBar = QProgressBar()
        self.progressBar.setRange(0, 100)
        self.progressBar.setValue(0)
        pb_layout.addWidget(self.progressLabel)
        pb_layout.addWidget(self.progressBar)
        self.progressBox.setVisible(True)
        mainFiltrationLayout.addWidget(self.progressBox)

        self.resultBox = QGroupBox("Filtration result")
        rb_layout = QVBoxLayout(self.resultBox)
        self.resultLabel = QLabel("No results yet.")
        self.resultDetail = QLabel("")
        self.resultDetail.setWordWrap(True)
        rb_layout.addWidget(self.resultLabel)
        rb_layout.addWidget(self.resultDetail)
        self.resultBox.setVisible(True)
        mainFiltrationLayout.addWidget(self.resultBox)
        self.closeAndSaveBtn = QPushButton("Close Window and Save Result")
        self.closeAndSaveBtn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.closeAndSaveBtn.clicked.connect(self.close)
        mainFiltrationLayout.addWidget(self.closeAndSaveBtn)

        self._thread = None
        self._worker = None
        self.filtered_df = None
        self.manually_adjusted = False   
        self._applying_preset = False    

    def _get_current_filtered_df(self):
        filtered_df = self.df.copy()
        for item in self.filters:
            min_value = self.spin_box[item]["min_spin_box"].value()
            max_value = self.spin_box[item]["max_spin_box"].value()
            filtered_df = filtered_df[(filtered_df[item] >= min_value) & (filtered_df[item] <= max_value)]
        return filtered_df

    def _refresh_current_library_ranges(self):
        for item in self.filters:
            min_value = self.original_ranges[item]["min"]
            max_value = self.original_ranges[item]["max"]
            if item in self.int_filters:
                self.spin_box[item]["range_label"].setText(f"{int(min_value)} - {int(max_value)}")
            else:
                self.spin_box[item]["range_label"].setText(f"{min_value:.2f} - {max_value:.2f}")

    def setFilterValues(self): 
        global filters_dictionary
        selected_filter = self.filterOptionsComboBox.currentText()
        if selected_filter == "Select":
            return
        else:
            self._applying_preset = True
            for item in self.filters:
                try:
                    self.spin_box[item]["min_spin_box"].setValue(max(filters_dictionary[selected_filter][item]["min"], self.spin_box[item]["min"]))
                except KeyError:
                    pass
                except Exception:
                    continue
                try:
                    self.spin_box[item]["max_spin_box"].setValue(min(filters_dictionary[selected_filter][item]["max"], self.spin_box[item]["max"]))
                except KeyError:
                    pass
                except Exception:
                    continue
            self.updateLabels()
            self._applying_preset = False
            self.manually_adjusted = False

    def resetFilters(self):
        for item in self.filters:
            self.spin_box[item]["min_spin_box"].setValue(self.spin_box[item]["min"])
            self.spin_box[item]["max_spin_box"].setValue(self.spin_box[item]["max"])
        self.filtered_df = None
        self.manually_adjusted = False
        self.filterOptionsComboBox.setCurrentText("Select")
        self.viewFilteredLibraryBtn.setDisabled(True)
        self.resultLabel.setText("No results yet.")
        self.resultDetail.setText("")
        self.progressLabel.setText("Waiting to start…")
        self.progressBar.setValue(0)
        self.resultBox.setVisible(True)
        self.preconfiguredFiltersGroupBox.setEnabled(True)
        for item in self.filters:
            for box in ["min_spin_box", "max_spin_box"]:
                self.spin_box[item][box].setEnabled(True)
        self.filterDfBtn.setEnabled(True)
        self.updateLabels()

    def updateLabels(self): 
        if not self._applying_preset:
            self.manually_adjusted = True
        for item in self.filters:
            min_value = self.spin_box[item]["min_spin_box"].value()
            max_value = self.spin_box[item]["max_spin_box"].value()
            self.spin_box[item]["min_value"] = min_value
            self.spin_box[item]["min_spin_box"].setMaximum(max_value)
            self.spin_box[item]["max_value"] = max_value
            self.spin_box[item]["max_spin_box"].setMinimum(min_value)
        self._refresh_current_library_ranges()

    def describeFilter(self, s):
        global filters_dictionary
        if s == "Select":
            self.filterDescription.setText("Description: ")
            self.applyPresetsBtn.setDisabled(True)
        else:
            try:
                self.filterDescription.setText("Description: "+filters_dictionary[s]["Info"])
                self.applyPresetsBtn.setDisabled(False)
                self.setFilterValues()
            except KeyError as exc:
                print(f"Error with filter or description: {exc}")
                self.applyPresetsBtn.setDisabled(True)
            except Exception as exc:
                print(f"Unexpected error with filter description: {exc}")
                self.applyPresetsBtn.setDisabled(True)

    def _current_filter_label(self):
        preset = self.filterOptionsComboBox.currentText()
        if preset != "Select" and not self.manually_adjusted:
            return preset
        return "Custom"

    def showFilteredLibraryTable(self):
        if getattr(self, "filtered_df", None) is None or self.filtered_df.empty:
            QMessageBox.information(self, "No filtered library", "No filtered molecules are available yet.")
            return
        
        display_df = _ensure_descriptor_columns(self.filtered_df.copy())

        if display_df.empty:
            QMessageBox.information(self, "No filtered library", "The filtered library is empty.")
            return
        
        preview_rows = min(100, len(display_df))
        display_df = display_df.head(preview_rows)
        dialog = QDialog(self)
        dialog.setWindowTitle("Filtered Library")
        dialog.resize(1300, 700)
        layout = QVBoxLayout(dialog)

        if len(self.filtered_df) > preview_rows:
            layout.addWidget(QLabel(f"Showing the first {preview_rows} of {len(self.filtered_df)} molecules to keep the view responsive."))

        table = QTableWidget()
        table.setAlternatingRowColors(True)
        table.setSortingEnabled(True)
        table.setRowCount(len(display_df))
        columns = ["Structure", "Name/ID", "SMILES"] + [col for col in DESCRIPTOR_COLUMNS if col in display_df.columns]
        table.setColumnCount(len(columns))
        table.setHorizontalHeaderLabels(columns)

        for table_row, (_, row) in enumerate(display_df.iterrows()):
            mol = _smiles_to_mol(row.get("SMILES"))
            structure_label = QLabel()
            structure_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            pixmap = _render_molecule_image(mol)
            if pixmap is not None:
                structure_label.setPixmap(pixmap)
            else:
                structure_label.setText("No structure")
            table.setCellWidget(table_row, 0, structure_label)

            name_value = row.get("Name") if "Name" in display_df.columns else ""
            if pd.isna(name_value):
                name_value = ""
            if not name_value and "ID" in display_df.columns:
                name_value = row.get("ID")
            if pd.isna(name_value):
                name_value = ""
            table.setItem(table_row, 1, QTableWidgetItem(str(name_value)))

            smiles_value = row.get("SMILES") if "SMILES" in display_df.columns else ""
            table.setItem(table_row, 2, QTableWidgetItem(str(smiles_value) if not pd.isna(smiles_value) else ""))
            for col_idx, col in enumerate([col for col in DESCRIPTOR_COLUMNS if col in display_df.columns], start=3):
                value = row.get(col)
                text = _format_table_value(value)
                table.setItem(table_row, col_idx, QTableWidgetItem(text))

        table.resizeColumnsToContents()
        layout.addWidget(table)
        dialog.exec()

    def start_filtration(self):
        if getattr(self, "df", None) is None or self.df.empty:
            QMessageBox.information(self, "No library", "Load or build a library before filtering.")
            return
        
        if self._thread is not None:
            try:
                if self._thread.isRunning():
                    QMessageBox.information(self, "Filtration running", "A filtration run is already in progress.")
                    return
            except RuntimeError:
                self._thread = None   
        self.preconfiguredFiltersGroupBox.setDisabled(True)

        for item in self.filters:
            for box in ["min_spin_box", "max_spin_box"]:
                self.spin_box[item][box].setDisabled(True)

        self.filterDfBtn.setDisabled(True)
        self.progressBox.setVisible(True)
        self.resultBox.setVisible(False)
        self.progressLabel.setText("Applying filters…")
        self.progressBar.setValue(0)
        self.viewFilteredLibraryBtn.setDisabled(True)
        self.resultLabel.setText("No results yet.")
        self.resultDetail.setText("")

        filter_ranges = {}
        
        for item in self.filters:
            filter_ranges[item] = {
                "min": self.spin_box[item]["min_spin_box"].value(),
                "max": self.spin_box[item]["max_spin_box"].value(),
                "original_min": self.spin_box[item]["min"],
                "original_max": self.spin_box[item]["max"],
            }

        filter_description = format_filter_ranges(filter_ranges)
        self.active_filter_description = filter_description
        self.active_filter_label = self._current_filter_label()
        
        self._thread = QThread()
        self._worker = FilterWorker(df=self.df, filter_ranges=filter_ranges, filter_label=self._current_filter_label())
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_worker_error)
        self._worker.error.connect(self._thread.quit)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.finished.connect(lambda: setattr(self, "_thread", None))
        self._worker.finished.connect(lambda: setattr(self, "_worker", None))
        
        self._thread.start()

    def _on_progress(self, value: int):
        self.progressBar.setValue(value)
        self.progressLabel.setText(f"Filtration running… {value}%")

    def _on_worker_error(self, message: str):
        self.preconfiguredFiltersGroupBox.setEnabled(True)

        for item in self.filters:
            for box in ["min_spin_box", "max_spin_box"]:
                self.spin_box[item][box].setEnabled(True)

        self.filterDfBtn.setEnabled(True)
        self.progressBox.setVisible(False)
        self.resultBox.setVisible(True)
        self.resultLabel.setText("Filtration failed")
        self.resultDetail.setText(message)
        self.viewFilteredLibraryBtn.setDisabled(True)
        self.filtered_df = None
        print(message)

    def _on_finished(self, payload: dict):
        self.preconfiguredFiltersGroupBox.setEnabled(True)

        for item in self.filters:
            for box in ["min_spin_box", "max_spin_box"]:
                self.spin_box[item][box].setEnabled(True)

        self.filterDfBtn.setEnabled(True)
        self.progressBox.setVisible(False)
        self.resultBox.setVisible(True)

        filters = payload.get("filters") or []
        fs = payload.get("final_size")
        kc = payload.get("key_characteristics", {})

        self.filtered_df = payload.get("filtered_df")

        if self.filtered_df is not None and not self.filtered_df.empty:
            self.viewFilteredLibraryBtn.setEnabled(True)
        else:
            self.viewFilteredLibraryBtn.setDisabled(True)

        chosen_str = ", ".join(filters) if filters else "None"
        self.resultLabel.setText("Filtration completed successfully")
        self.resultDetail.setText(
            f"<b>Filters:</b> {chosen_str}<br>"
            f"<b>Final library size:</b> {fs} molecules<br>"
            f"<b>Key characteristics:</b> "
            f"avg_MW={kc.get('avg_MW','?')} | "
            f"avg_logP={kc.get('avg_logP','?')} | "
            f"avg_SAScore={kc.get('avg_SAScore','?')} | "
            f"aromatic_rings≥2=%{kc.get('aromatic_rings≥2_%','?')}"
        )

    def pushResultsToMain(self):
        global mainmenu
        global baseSMILES
        if self.filtered_df is None:
            return
        mainmenu.library_df = _ensure_descriptor_columns(self.filtered_df.copy())
        baseSMILES = DeDuplicate(self.filtered_df["SMILES"].tolist())

        if self.active_filter_description:
            filter_text = (
                f"{self.active_filter_label}: "
                f"{self.active_filter_description}"
            )
        else:
            filter_text = self.active_filter_label

        mainmenu.log_message(
            f"Property filtration applied [{filter_text}]: "
            f"{len(baseSMILES):,} molecule(s) retained."
        )
        mainmenu.refreshFiltrationAvailability()

    def closeEvent(self, event: QCloseEvent):
        self.pushResultsToMain()
        event.accept()


class SimilarityWorker(QObject):
    progress = pyqtSignal(int) 
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)    

    def __init__(self, df, ref_smiles_list, ref_name, agg_method):
        super().__init__()
        self.df = df                        
        self.ref_smiles_list = ref_smiles_list 
        self.ref_name = ref_name    
        self.agg_method = agg_method

    def run(self):
        try:
            if self.df is None or self.df.empty:
                self.error.emit("No library is loaded to score.")
                return
            ref_fps = []
            for ref_smi in self.ref_smiles_list:
                ref_mol = _smiles_to_mol(ref_smi)
                fp = _morgan_fp(ref_mol)
                if fp is not None:
                    ref_fps.append(fp)
            if not ref_fps:
                self.error.emit("No valid reference structure(s) could be parsed.")
                return
            self.progress.emit(5)

            df = self.df.copy()
            smiles_list = df["SMILES"].tolist()
            n = max(1, len(smiles_list))
            tanimoto_scores = []
            for i, smi in enumerate(smiles_list):
                mol = _smiles_to_mol(smi)
                fp = _morgan_fp(mol)
                if fp is None:
                    tanimoto_scores.append(None)
                else:
                    sims = [DataStructs.TanimotoSimilarity(fp, ref_fp) for ref_fp in ref_fps]
                    tanimoto_scores.append(max(sims) if self.agg_method == "max" else (sum(sims) / len(sims)))
                if i % max(1, n // 20) == 0:
                    self.progress.emit(5 + int(90 * i / n))

            df["TanimotoSimilarity"] = tanimoto_scores
            df["SimilarityPercent"] = [None if s is None else round(s * 100, 1) for s in tanimoto_scores]

            ref_count = len(ref_fps)
            method_phrase = "highest (best-match)" if self.agg_method == "max" else "average"
            def _describe(score):
                if score is None:
                    return None
                if ref_count == 1:
                    return f"The molecule has {round(score, 3)} similarity score to the reference {self.ref_name}."
                return (f"The molecule has {round(score, 3)} {method_phrase} similarity score to the "
                        f"reference set {self.ref_name} ({ref_count} molecules).")
            df["SimilarityDescription"] = [_describe(s) for s in tanimoto_scores]

            self.progress.emit(100)
            self.finished.emit({
                "scored_df": df,
                "ref_count": ref_count,
                "agg_method": self.agg_method,
                "ref_name": self.ref_name,
            })
        except Exception as exc:
            self.error.emit(f"Similarity calculation failed: {exc}")

class ChemicalSpaceWorker(QObject):
    progress = pyqtSignal(int)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, library_smiles, ref_smiles, lib_name, ref_name):
        super().__init__()
        self.library_smiles = library_smiles
        self.ref_smiles = ref_smiles
        self.lib_name = lib_name
        self.ref_name = ref_name

    def run(self):
        try:
            if not self.library_smiles:
                self.error.emit("No library is loaded to plot.")
                return
            lib_X, _lib_valid = _fingerprint_matrix(self.library_smiles)
            self.progress.emit(35)
            if lib_X.shape[0] < 2:
                self.error.emit("Need at least 2 valid library structures to plot chemical space.")
                return

            ref_X = None
            if self.ref_smiles:
                ref_X, _ref_valid = _fingerprint_matrix(self.ref_smiles)
                if ref_X.shape[0] == 0:
                    ref_X = None 
            self.progress.emit(60)

            combined = np.vstack([lib_X, ref_X]) if ref_X is not None and ref_X.shape[0] else lib_X
            coords, explained = _pca_2d(combined)
            self.progress.emit(90)

            lib_coords = coords[:lib_X.shape[0]]
            ref_coords = coords[lib_X.shape[0]:] if ref_X is not None and ref_X.shape[0] else None

            self.progress.emit(100)
            self.finished.emit({
                "lib_coords": lib_coords,
                "ref_coords": ref_coords,
                "explained": explained,
                "lib_name": self.lib_name,
                "ref_name": self.ref_name,
            })
        except Exception as exc:
            self.error.emit(f"Chemical-space calculation failed: {exc}")

def _load_known_aggregators() -> list:
    if not os.path.isfile(_KNOWN_AGGREGATORS_PATH):
        return []
    with open(_KNOWN_AGGREGATORS_PATH) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

class AggregationWorker(QObject):
    progress = pyqtSignal(int)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, library_smiles, aggregator_smiles):
        super().__init__()
        self.library_smiles = library_smiles
        self.aggregator_smiles = aggregator_smiles

    def run(self):
        try:
            if not self.aggregator_smiles:
                self.error.emit("No known-aggregators reference file found (known_aggregators.smi).")
                return
            if not self.library_smiles:
                self.error.emit("No library is loaded to screen.")
                return

            agg_fps = []
            for smi in self.aggregator_smiles:
                fp = _morgan_fp(_smiles_to_mol(smi))
                if fp is not None:
                    agg_fps.append(fp)
            self.progress.emit(15)
            if not agg_fps:
                self.error.emit("The known-aggregators reference file contained no valid structures.")
                return

            rows = []
            total = max(1, len(self.library_smiles))
            for idx, smi in enumerate(self.library_smiles):
                mol = _smiles_to_mol(smi)
                if mol is None:
                    continue
                fp = _morgan_fp(mol)
                max_tc = max((DataStructs.TanimotoSimilarity(fp, afp) for afp in agg_fps), default=0.0) if fp is not None else 0.0
                logp = Descriptors.MolLogP(mol)
                if logp > 3 and max_tc >= 0.85:
                    risk = "High"
                elif logp > 3 or max_tc >= 0.85:
                    risk = "Medium"
                else:
                    risk = "Low"
                rows.append({"SMILES": smi, "logP": round(logp, 2), "MaxTanimotoToAggregator": round(max_tc, 3), "AggregationRisk": risk})
                if idx % max(1, total // 100) == 0:
                    self.progress.emit(15 + int(80 * idx / total))

            self.progress.emit(100)
            result_df = pd.DataFrame(rows)
            counts = result_df["AggregationRisk"].value_counts().to_dict() if not result_df.empty else {}
            self.finished.emit({"result_df": result_df, "counts": counts})
        except Exception as exc:
            self.error.emit(f"Aggregation screening failed: {exc}")

class AggregationResultsDialog(QDialog):

    def __init__(self, parent, result_df, counts):
        super().__init__(parent)
        self.setWindowTitle("Aggregation Risk Screen")
        self.resize(420, 260)
        self.result_df = result_df
        layout = QVBoxLayout(self)

        summary = QLabel(
            f"High risk: {counts.get('High', 0)}\n"
            f"Medium risk: {counts.get('Medium', 0)}\n"
            f"Low risk: {counts.get('Low', 0)}\n\n"
            "High = LogP > 3 AND Tanimoto >= 0.85 to a known aggregator\n"
            "Medium = either condition alone.  Low = neither.\n"
            "(Irwin et al., J Med Chem 2015, 58, 7076-7087)"
        )
        layout.addWidget(summary)

        saveBtn = QPushButton("Save Full Results as CSV")
        saveBtn.setCursor(Qt.CursorShape.PointingHandCursor)
        saveBtn.clicked.connect(self.saveCsv)
        layout.addWidget(saveBtn)

        closeBtn = QPushButton("Close")
        closeBtn.clicked.connect(self.close)
        layout.addWidget(closeBtn)

    def saveCsv(self):
        global workingDirectory
        path, _selected_filter = QFileDialog.getSaveFileName(
            parent=self,
            caption="Save aggregation screen results",
            directory=os.path.join(workingDirectory, "aggregation_screen.csv") if workingDirectory else "aggregation_screen.csv",
            filter="CSV (*.csv)"
        )
        if not path:
            return
        if not path.lower().endswith(".csv"):
            path += ".csv"
        try:
            self.result_df.to_csv(path, index=False)
            workingDirectory = os.path.dirname(path)
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", f"Could not save results: {exc}")

class ChemicalSpaceDialog(QDialog):

    def __init__(self, parent, lib_coords, ref_coords, explained, lib_name, ref_name):
        super().__init__(parent)
        self.setWindowTitle("Chemical Space (PCA)")
        self.resize(700, 700)
        self.lib_coords = lib_coords
        self.ref_coords = ref_coords
        self.explained = explained
        layout = QVBoxLayout(self)

        self.figure = Figure(figsize=(6, 5))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        customGroupBox = QGroupBox("Customize Labels")
        customLayout = QFormLayout(customGroupBox)
        self.titleBox = QLineEdit("Chemical Space - PCA on Morgan Fingerprints")
        self.xLabelBox = QLineEdit(f"PC1 ({explained[0] * 100:.1f}% variance)")
        self.yLabelBox = QLineEdit(f"PC2 ({explained[1] * 100:.1f}% variance)")
        self.libLabelBox = QLineEdit(lib_name)
        self.refLabelBox = QLineEdit(ref_name or "Reference")
        self.refLabelBox.setEnabled(ref_coords is not None and len(ref_coords) > 0)
        customLayout.addRow("Plot title:", self.titleBox)
        customLayout.addRow("X-axis label:", self.xLabelBox)
        customLayout.addRow("Y-axis label:", self.yLabelBox)
        customLayout.addRow("Library legend label:", self.libLabelBox)
        customLayout.addRow("Reference legend label:", self.refLabelBox)
        applyBtn = QPushButton("Apply Labels")
        applyBtn.setCursor(Qt.CursorShape.PointingHandCursor)
        applyBtn.clicked.connect(self._redraw)
        customLayout.addRow(applyBtn)
        layout.addWidget(customGroupBox)

        btnRow = QHBoxLayout()
        saveBtn = QPushButton("Save as PNG")
        saveBtn.setCursor(Qt.CursorShape.PointingHandCursor)
        saveBtn.clicked.connect(self.savePlot)
        btnRow.addWidget(saveBtn)

        closeBtn = QPushButton("Close")
        closeBtn.clicked.connect(self.close)
        btnRow.addWidget(closeBtn)
        layout.addLayout(btnRow)

        self._redraw()

    def _redraw(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        has_ref = self.ref_coords is not None and len(self.ref_coords) > 0
        lib_kwargs = dict(s=18, alpha=0.6, label=self.libLabelBox.text() or "Library",
                          color="#4E5659", marker="o")
        ref_kwargs = dict(s=32, alpha=0.85, label=self.refLabelBox.text() or "Reference",
                          color="#2E86AB", marker="o")
        if has_ref and len(self.ref_coords) < len(self.lib_coords):
            ax.scatter(self.lib_coords[:, 0], self.lib_coords[:, 1], zorder=1, **lib_kwargs)
            ax.scatter(self.ref_coords[:, 0], self.ref_coords[:, 1], zorder=2, **ref_kwargs)
        elif has_ref:
            ax.scatter(self.ref_coords[:, 0], self.ref_coords[:, 1], zorder=1, **ref_kwargs)
            ax.scatter(self.lib_coords[:, 0], self.lib_coords[:, 1], zorder=2, **lib_kwargs)
        else:
            ax.scatter(self.lib_coords[:, 0], self.lib_coords[:, 1], zorder=1, **lib_kwargs)
        ax.set_xlabel(self.xLabelBox.text())
        ax.set_ylabel(self.yLabelBox.text())
        ax.set_title(self.titleBox.text())
        ax.legend()
        self.figure.tight_layout()
        self.canvas.draw()

    def savePlot(self):
        global workingDirectory
        path, _selected_filter = QFileDialog.getSaveFileName(
            parent=self,
            caption="Save chemical space plot",
            directory=os.path.join(workingDirectory, "chemical_space.png") if workingDirectory else "chemical_space.png",
            filter="PNG Image (*.png)"
        )
        if not path:
            return
        if not path.lower().endswith(".png"):
            path += ".png"
        try:
            self.figure.savefig(path, dpi=300, bbox_inches="tight")
            workingDirectory = os.path.dirname(path)
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", f"Could not save plot: {exc}")

class SimilarityWindow(QWidget):

    def __init__(self, parent=None, df=None):
        super().__init__(parent)
        self.df = _ensure_descriptor_columns(df.copy()) if df is not None else pd.DataFrame(columns=["SMILES"])
        self.scored_df = None  
        self.filtered_df = None
        self.ref_smiles = []   
        self.ref_name = ""     
        self._thread = None
        self._worker = None

        self.setWindowTitle("Similarity Scoring & Chemical Space")
        self.resize(760, 480)
        layout = QVBoxLayout(self)

        self.importGroupBox = QGroupBox("Reference Molecule or Database")
        importLayout = QGridLayout(self.importGroupBox)
        self.importRefBtn = QPushButton("Import Reference Molecule or Database")
        self.importRefBtn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.importRefBtn.clicked.connect(self.importReference)
        self.refInfoLabel = QLabel("No reference imported yet. Accepts a single-molecule .sdf/.mol file\n"
                                    "(one reference) or a multi-molecule .sdf file (a reference set).")
        importLayout.addWidget(self.importRefBtn, 0, 0)
        importLayout.addWidget(self.refInfoLabel, 0, 1)
        layout.addWidget(self.importGroupBox)

        self.methodGroupBox = QGroupBox("Reference-Set Aggregation")
        methodLayout = QVBoxLayout(self.methodGroupBox)
        methodLayout.addWidget(QLabel("This reference is a set of molecules. Combine per-reference Tanimoto scores using:"))
        self.methodComboBox = QComboBox()
        self.methodComboBox.addItems([
            "Best match (max Tanimoto to any single reference molecule) - recommended",
            "Average (mean Tanimoto across the whole reference set)",
        ])
        methodLayout.addWidget(self.methodComboBox)
        self.methodGroupBox.setLayout(methodLayout)
        self.methodGroupBox.setVisible(False)
        layout.addWidget(self.methodGroupBox)

        self.calcBtn = QPushButton("Calculate Similarity Score")
        self.calcBtn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.calcBtn.clicked.connect(self.start_calculation)
        self.calcBtn.setDisabled(True)   
        layout.addWidget(self.calcBtn)

        self.plotSpaceBtn = QPushButton("Plot Chemical Space")
        self.plotSpaceBtn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.plotSpaceBtn.clicked.connect(self.start_plot_chemical_space)
        layout.addWidget(self.plotSpaceBtn)

        self.aggScreenBtn = QPushButton("Screen for Aggregators")
        self.aggScreenBtn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.aggScreenBtn.clicked.connect(self.start_aggregation_screen)
        layout.addWidget(self.aggScreenBtn)

        self.progressBar = QProgressBar()
        self.progressBar.setVisible(False)
        layout.addWidget(self.progressBar)

        self.resultsGroupBox = QGroupBox("Similarity Results")
        resultsLayout = QGridLayout(self.resultsGroupBox)
        self.resultsSummaryLabel = QLabel("No scores calculated yet.")
        self.resultsSummaryLabel.setWordWrap(True)
        resultsLayout.addWidget(self.resultsSummaryLabel, 0, 0, 1, 4)

        resultsLayout.addWidget(QLabel("Keep molecules with similarity between:"), 1, 0)
        self.minPercentSpin = QDoubleSpinBox()
        self.minPercentSpin.setRange(0, 100)
        self.minPercentSpin.setDecimals(1)
        self.minPercentSpin.setValue(0.0)
        self.minPercentSpin.setSuffix(" %")
        self.maxPercentSpin = QDoubleSpinBox()
        self.maxPercentSpin.setRange(0, 100)
        self.maxPercentSpin.setDecimals(1)
        self.maxPercentSpin.setValue(100.0)
        self.maxPercentSpin.setSuffix(" %")
        resultsLayout.addWidget(self.minPercentSpin, 1, 1)
        resultsLayout.addWidget(QLabel("and"), 1, 2)
        resultsLayout.addWidget(self.maxPercentSpin, 1, 3)

        self.applyFilterBtn = QPushButton("Apply Similarity Filter to Library")
        self.applyFilterBtn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.applyFilterBtn.clicked.connect(self.applySimilarityFilter)
        self.applyFilterBtn.setDisabled(True)
        resultsLayout.addWidget(self.applyFilterBtn, 2, 0, 1, 2)

        self.viewResultsBtn = QPushButton("View Scored Library")
        self.viewResultsBtn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.viewResultsBtn.clicked.connect(self.showScoredLibraryTable)
        self.viewResultsBtn.setDisabled(True)
        resultsLayout.addWidget(self.viewResultsBtn, 2, 2, 1, 2)

        self.filterResultLabel = QLabel("")
        self.filterResultLabel.setWordWrap(True)
        resultsLayout.addWidget(self.filterResultLabel, 3, 0, 1, 4)

        self.resultsGroupBox.setLayout(resultsLayout)
        self.resultsGroupBox.setVisible(False)
        layout.addWidget(self.resultsGroupBox)

        self.closeAndSaveBtn = QPushButton("Close Window and Save Result")
        self.closeAndSaveBtn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.closeAndSaveBtn.clicked.connect(self.close)
        layout.addWidget(self.closeAndSaveBtn)

        self._refresh_action_buttons()   

    def _refresh_action_buttons(self):
        plotting = getattr(self, "_plot_thread", None) is not None
        scoring = self._thread is not None
        screening = getattr(self, "_agg_thread", None) is not None
        busy = plotting or scoring or screening
        has_library = self.df is not None and not self.df.empty
        has_reference = bool(self.ref_smiles)
        self.calcBtn.setEnabled(has_reference and not busy)
        self.plotSpaceBtn.setEnabled(MATPLOTLIB_AVAILABLE and has_library and not busy)
        self.aggScreenBtn.setEnabled(has_library and not busy)

        if not MATPLOTLIB_AVAILABLE:
            self.plotSpaceBtn.setToolTip(f"Chemical-space plotting is disabled: {MATPLOTLIB_IMPORT_ERROR}")
        else:
            self.plotSpaceBtn.setToolTip("Plots the library alone, or overlaid with the reference if one is imported.")

    def importReference(self):
        global workingDirectory

        file_filter = 'SD File (*.sdf);;Mol File (*.mol)'
        path, _selected_filter = QFileDialog.getOpenFileName(
            parent=self,
            caption='Select reference molecule or reference database',
            directory=workingDirectory,
            filter=file_filter,
            initialFilter='SD File (*.sdf)'
        )
        if path:
            workingDirectory = os.path.dirname(path)
        if not path:
            return
        smis = []
        try:
            if path.lower().endswith(".sdf"):
                threads = max(1, int(os.cpu_count()) - 2)
                with Chem.MultithreadedSDMolSupplier(path, numWriterThreads=threads) as suppl:
                    for mol in suppl:
                        if mol is None:
                            continue
                        smis.append(Chem.MolToSmiles(mol))
            else:
                mol = Chem.MolFromMolFile(path)
                if mol is not None:
                    smis.append(Chem.MolToSmiles(mol))
        except Exception as exc:
            QMessageBox.critical(self, "Import failed", f"Could not read reference file: {exc}")
            return
        if not smis:
            QMessageBox.warning(self, "No structures found", "No valid structure(s) could be parsed from that file.")
            return

        self.ref_smiles = DeDuplicate(smis)
        self.ref_name = os.path.splitext(os.path.basename(path))[0]
        if len(self.ref_smiles) == 1:
            self.refInfoLabel.setText(f"Reference molecule loaded: \"{self.ref_name}\" (1 structure).")
            self.methodGroupBox.setVisible(False)
        else:
            self.refInfoLabel.setText(
                f"Reference database loaded: \"{self.ref_name}\" ({len(self.ref_smiles)} structures after de-duplication)."
            )
            self.methodGroupBox.setVisible(True)

        self.scored_df = None
        self.filtered_df = None
        self.resultsGroupBox.setVisible(False)
        self._refresh_action_buttons()

    def start_calculation(self):
        if self.df is None or self.df.empty:
            QMessageBox.information(self, "No library", "There is no library loaded to score.")
            return
        if not self.ref_smiles:
            QMessageBox.information(self, "No reference", "Import a reference molecule or database first.")
            return
        if self._thread is not None:
            try:
                if self._thread.isRunning():
                    QMessageBox.information(self, "Calculation running", "A similarity calculation is already in progress.")
                    return
            except RuntimeError:
                self._thread = None

        agg_method = "max" if self.methodComboBox.currentIndex() == 0 else "mean"
        self.importGroupBox.setDisabled(True)
        self.methodGroupBox.setDisabled(True)
        self.calcBtn.setDisabled(True)
        self.plotSpaceBtn.setDisabled(True)
        self.aggScreenBtn.setDisabled(True)
        self.progressBar.setVisible(True)
        self.progressBar.setValue(0)
        self.resultsGroupBox.setVisible(False)

        self._thread = QThread()
        self._worker = SimilarityWorker(df=self.df, ref_smiles_list=self.ref_smiles, ref_name=self.ref_name, agg_method=agg_method)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self.progressBar.setValue)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_worker_error)
        self._worker.error.connect(self._thread.quit)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.finished.connect(lambda: setattr(self, "_thread", None))
        self._worker.finished.connect(lambda: setattr(self, "_worker", None))

        self._thread.start()

    def _on_worker_error(self, message: str):
        self.importGroupBox.setEnabled(True)
        self.methodGroupBox.setEnabled(True)
        self._refresh_action_buttons()
        self.progressBar.setVisible(False)
        QMessageBox.critical(self, "Calculation failed", message)
        print(message)

    def _on_finished(self, payload: dict):
        self.importGroupBox.setEnabled(True)
        self.methodGroupBox.setEnabled(True)
        self._refresh_action_buttons()
        self.progressBar.setVisible(False)

        self.scored_df = payload.get("scored_df")
        self.filtered_df = None
        ref_count = payload.get("ref_count", 1)
        agg_method = payload.get("agg_method", "max")
        if self.scored_df is None or self.scored_df.empty:
            QMessageBox.information(self, "No results", "No similarity scores could be computed.")
            return

        valid = self.scored_df["TanimotoSimilarity"].dropna()
        if len(valid):
            summary = (
                f"Scored {len(self.scored_df)} molecule(s) against {ref_count} reference structure(s) "
                f"using the {'best-match' if agg_method == 'max' else 'average'} method.<br>"
                f"Similarity range: {valid.min() * 100:.1f}% - {valid.max() * 100:.1f}% "
                f"(library mean: {valid.mean() * 100:.1f}%)."
            )
        else:
            summary = "Scoring finished, but no molecule in the library could be compared (all failed to parse)."
        self.resultsSummaryLabel.setText(summary)
        self.resultsGroupBox.setVisible(True)
        self.applyFilterBtn.setEnabled(True)
        self.viewResultsBtn.setEnabled(True)
        self.filterResultLabel.setText("")

    def start_plot_chemical_space(self):
        if self.df is None or self.df.empty:
            QMessageBox.information(self, "No library", "There is no library loaded to plot.")
            return
        if getattr(self, "_plot_thread", None) is not None:
            try:
                if self._plot_thread.isRunning():
                    QMessageBox.information(self, "Plot running", "A chemical-space calculation is already in progress.")
                    return
            except RuntimeError:
                self._plot_thread = None

        self.plotSpaceBtn.setDisabled(True)
        self.calcBtn.setDisabled(True)
        self.aggScreenBtn.setDisabled(True)
        self.progressBar.setVisible(True)
        self.progressBar.setValue(0)

        self._plot_thread = QThread()
        self._plot_worker = ChemicalSpaceWorker(
            library_smiles=self.df["SMILES"].tolist(),
            ref_smiles=list(self.ref_smiles),   
            lib_name=(name if name else "Current Library"),
            ref_name=self.ref_name,
        )
        self._plot_worker.moveToThread(self._plot_thread)

        self._plot_thread.started.connect(self._plot_worker.run)
        self._plot_worker.progress.connect(self.progressBar.setValue)
        self._plot_worker.finished.connect(self._on_plot_finished)
        self._plot_worker.error.connect(self._on_plot_error)
        self._plot_worker.error.connect(self._plot_thread.quit)
        self._plot_worker.finished.connect(self._plot_thread.quit)
        self._plot_worker.finished.connect(self._plot_worker.deleteLater)
        self._plot_thread.finished.connect(self._plot_thread.deleteLater)
        self._plot_thread.finished.connect(lambda: setattr(self, "_plot_thread", None))
        self._plot_worker.finished.connect(lambda: setattr(self, "_plot_worker", None))

        self._plot_thread.start()

    def _on_plot_error(self, message: str):
        self._refresh_action_buttons()
        self.progressBar.setVisible(False)
        QMessageBox.critical(self, "Plot failed", message)
        print(message)

    def _on_plot_finished(self, payload: dict):
        self._refresh_action_buttons()
        self.progressBar.setVisible(False)
        dialog = ChemicalSpaceDialog(
            parent=self,
            lib_coords=payload["lib_coords"],
            ref_coords=payload.get("ref_coords"),
            explained=payload["explained"],
            lib_name=payload.get("lib_name", "Current Library"),
            ref_name=payload.get("ref_name"),
        )
        dialog.exec()

    def start_aggregation_screen(self):
        if self.df is None or self.df.empty:
            QMessageBox.information(self, "No library", "There is no library loaded to screen.")
            return
        aggregator_path = ensure_known_aggregators_file()
        if aggregator_path is None:
            QMessageBox.warning(
                self, "Aggregator reference unavailable",
                "MOLDE could not obtain the Aggregator Advisor reference data. "
                "Check your internet connection and try again."
            )
            return

        aggregators = _load_known_aggregators()
        if not aggregators:
            QMessageBox.warning(
                self, "Invalid aggregator reference",
                "known_aggregators.smi exists but contains no usable SMILES. "
                "Delete the file and run the screen again so MOLDE can recreate it."
            )
            return

        self.aggScreenBtn.setDisabled(True)
        self.calcBtn.setDisabled(True)
        self.plotSpaceBtn.setDisabled(True)
        self.progressBar.setVisible(True)
        self.progressBar.setValue(0)

        self._agg_thread = QThread()
        self._agg_worker = AggregationWorker(self.df["SMILES"].tolist(), aggregators)
        self._agg_worker.moveToThread(self._agg_thread)

        self._agg_thread.started.connect(self._agg_worker.run)
        self._agg_worker.progress.connect(self.progressBar.setValue)
        self._agg_worker.finished.connect(self._on_aggregation_finished)
        self._agg_worker.error.connect(self._on_aggregation_error)
        self._agg_worker.error.connect(self._agg_thread.quit)
        self._agg_worker.finished.connect(self._agg_thread.quit)
        self._agg_worker.finished.connect(self._agg_worker.deleteLater)
        self._agg_thread.finished.connect(self._agg_thread.deleteLater)
        self._agg_thread.finished.connect(lambda: setattr(self, "_agg_thread", None))
        self._agg_worker.finished.connect(lambda: setattr(self, "_agg_worker", None))

        self._agg_thread.start()

    def _on_aggregation_error(self, message: str):
        self._refresh_action_buttons()
        self.progressBar.setVisible(False)
        QMessageBox.critical(self, "Screening failed", message)
        print(message)

    def _on_aggregation_finished(self, payload: dict):
        self._refresh_action_buttons()
        self.progressBar.setVisible(False)
        dialog = AggregationResultsDialog(self, payload["result_df"], payload["counts"])
        dialog.exec()

    def applySimilarityFilter(self):
        if self.scored_df is None or self.scored_df.empty:
            QMessageBox.information(self, "No scores", "Calculate similarity scores before filtering.")
            return
        
        lo_pct = self.minPercentSpin.value()
        hi_pct = self.maxPercentSpin.value()

        if lo_pct > hi_pct:
            QMessageBox.warning(self, "Invalid range", "Minimum similarity cannot exceed maximum similarity.")
            return
        
        df = self.scored_df
        mask = df["SimilarityPercent"].notna() & (df["SimilarityPercent"] >= lo_pct) & (df["SimilarityPercent"] <= hi_pct)
        self.filtered_df = df[mask].copy()

        self.filterResultLabel.setText(
            f"Filtered to {len(self.filtered_df)} of {len(df)} molecule(s) with similarity between "
            f"{lo_pct:.1f}% and {hi_pct:.1f}%. This will become the library's new state once this window is closed."
        )

        self.viewResultsBtn.setEnabled(True)

    def showScoredLibraryTable(self):
        display_source = self.filtered_df if self.filtered_df is not None else self.scored_df

        if display_source is None or display_source.empty:
            QMessageBox.information(self, "No results", "No scored molecules are available yet.")
            return
        
        display_df = display_source.copy()
        preview_rows = min(100, len(display_df))
        display_df = display_df.head(preview_rows)

        dialog = QDialog(self)
        dialog.setWindowTitle("Scored Library" if self.filtered_df is None else "Similarity-Filtered Library")
        dialog.resize(1300, 700)
        layout = QVBoxLayout(dialog)
        if len(display_source) > preview_rows:
            layout.addWidget(QLabel(f"Showing the first {preview_rows} of {len(display_source)} molecules to keep the view responsive."))

        table = QTableWidget()
        table.setAlternatingRowColors(True)
        table.setSortingEnabled(True)
        table.setRowCount(len(display_df))
        columns = ["Structure", "Name/ID", "SMILES", "Similarity %"] + [col for col in DESCRIPTOR_COLUMNS if col in display_df.columns]
        table.setColumnCount(len(columns))
        table.setHorizontalHeaderLabels(columns)

        for table_row, (_, row) in enumerate(display_df.iterrows()):
            mol = _smiles_to_mol(row.get("SMILES"))
            structure_label = QLabel()
            structure_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            pixmap = _render_molecule_image(mol)
            if pixmap is not None:
                structure_label.setPixmap(pixmap)
            else:
                structure_label.setText("No structure")
            table.setCellWidget(table_row, 0, structure_label)

            name_value = row.get("Name") if "Name" in display_df.columns else ""
            if pd.isna(name_value):
                name_value = ""
            if not name_value and "ID" in display_df.columns:
                name_value = row.get("ID")
            if pd.isna(name_value):
                name_value = ""
            table.setItem(table_row, 1, QTableWidgetItem(str(name_value)))

            smiles_value = row.get("SMILES") if "SMILES" in display_df.columns else ""
            table.setItem(table_row, 2, QTableWidgetItem(str(smiles_value) if not pd.isna(smiles_value) else ""))

            table.setItem(table_row, 3, QTableWidgetItem(_format_table_value(row.get("SimilarityPercent"))))

            for col_idx, col in enumerate([col for col in DESCRIPTOR_COLUMNS if col in display_df.columns], start=4):
                table.setItem(table_row, col_idx, QTableWidgetItem(_format_table_value(row.get(col))))

        table.resizeColumnsToContents()
        layout.addWidget(table)
        dialog.exec()

    def closeEvent(self, event: QCloseEvent):
        self.pushResultsToMain()
        event.accept()

    def pushResultsToMain(self):
        global mainmenu
        global baseSMILES

        result_df = self.filtered_df if self.filtered_df is not None else self.scored_df

        if result_df is None:
            return
        
        mainmenu.library_df = _ensure_descriptor_columns(result_df.copy())
        baseSMILES = DeDuplicate(result_df["SMILES"].tolist())

        if self.filtered_df is not None:
            mainmenu.log_message(
                f"Similarity filtration complete: "
                f"{len(baseSMILES):,} molecule(s) retained at "
                f"{self.minPercentSpin.value():.1f}–"
                f"{self.maxPercentSpin.value():.1f}% similarity to "
                f"\"{self.ref_name}\"."
            )
        else:
            mainmenu.log_message(
                f"Similarity scoring complete: "
                f"{len(baseSMILES):,} molecule(s) scored against "
                f"\"{self.ref_name}\"."
            )

        mainmenu.refreshFiltrationAvailability()


class Main(QWidget):
    global oneReactions, twoReactions, reactions_dictionary, reaction, startup_log, baseFiles, baseSMILES, reactantSMILES, abbaVar, smarts, smis_a, smis_b, rxn_products, name, viewerList
    
    def __init__(self):
        super().__init__()
        self.createImportMoleculesGroupBox()
        self.createSelectReactionGroupBox()
        self.createMenuBar()
        self.createRunGroupBox()
        self.createVisualizationGroupBox()
        self.createMiddleGroupBox()
        self.createExportGroupBox()
        self.createManageLibraryGroupBox()
        self.createFiltrationGroupBox()

        QApplication.instance().aboutToQuit.connect(self.cleanupBeforeQuit)

        self.active_filter_labels = set()
        self.second = None
        self.third = None 
        self._exportThread = None
        self._exportWorker = None

        self._substr_thread = None
        self._substr_worker = None

        self._descriptor_thread = None
        self._descriptor_worker = None

        self.nameBoxLabel = QLabel("Library Name:")
        self.libraryNameBox = QLineEdit("")
        self.libraryNameBox.textEdited.connect(self.activateNameBtn)
        self.libraryNameBox.returnPressed.connect(self.setName)
        self.libraryNameBtn = QPushButton("Set Name")
        self.libraryNameBtn.clicked.connect(self.setName)
        nameBoxLayout = QGridLayout()
        nameBoxLayout.addWidget(self.nameBoxLabel, 0, 0)
        nameBoxLayout.addWidget(self.libraryNameBox, 0, 1)
        nameBoxLayout.addWidget(self.libraryNameBtn, 0, 2)

        topLayout = QGridLayout()
        topLayout.addWidget(self.importMoleculesGroupBox, 0, 0)
        topLayout.addWidget(self.selectReactionGroupBox, 0, 1)
        topLayout.addWidget(self.runGroupBox, 0, 2)
        topLayout.addWidget(self.visualizationGroupBox, 0, 3)
        for box in (self.importMoleculesGroupBox, self.selectReactionGroupBox,
                    self.runGroupBox, self.visualizationGroupBox):
            box.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        bottomLayout = QGridLayout()
        bottomLayout.addWidget(self.filtrationGroupBox, 0, 0)
        bottomLayout.addWidget(self.exportGroupBox, 0, 1)
        bottomLayout.addWidget(self.manageLibraryGroupBox, 0, 2)
        for box in (self.filtrationGroupBox, self.exportGroupBox, self.manageLibraryGroupBox):
            box.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        mainLayout = QGridLayout()
        mainLayout.addWidget(self.menuBar, 0, 0)
        mainLayout.addLayout(nameBoxLayout, 1, 0)
        mainLayout.addLayout(topLayout, 2, 0)
        mainLayout.addWidget(self.middleGroupBox, 3, 0)
        mainLayout.addLayout(bottomLayout, 4, 0)
        self.setLayout(mainLayout)

        self.importMoleculesGroupBox.setDisabled(True)
        self.selectReactionGroupBox.setDisabled(True)
        self.runGroupBox.setDisabled(True)
        self.exportGroupBox.setDisabled(True)
        self.visualizationGroupBox.setDisabled(True)
        self.filterLibraryBtn.setDisabled(True)
        self.similarityScoreBtn.setDisabled(True)
        self.mergeDatabaseBtn.setDisabled(True)
        self.splitDatabaseBtn.setDisabled(True)
        self.filterSubstrBtn.setDisabled(True)
        self.libraryNameBtn.setDisabled(True)
        self.retainStartingCheckBox.setDisabled(True)
        self.reactionHasRun = False

        self.setWindowTitle("MOLDE")

    def cleanupBeforeQuit(self):
        if getattr(self, "second", None) is not None:
            self.second.close()
        if getattr(self, "third", None) is not None:
            self.third.close()
        if getattr(self, "w", None) is not None:
            self.w.close()

    def createFiltrationGroupBox(self):
        self.filtrationGroupBox = QGroupBox("Filtration")

        self.filterLibraryBtn = QPushButton("Filter Library by Property")
        self.filterLibraryBtn.clicked.connect(self.on_filter_clicked)

        self.similarityScoreBtn = QPushButton("Similarity Scoring && Chemical Space")
        self.similarityScoreBtn.clicked.connect(self.open_similarity_window)

        self.filterSubstrBtn = QPushButton("Filter Library by Substructure")
        self.filterSubstrBtn.clicked.connect(self.on_filter_substr_clicked)

        for btn in (
            self.filterLibraryBtn,
            self.filterSubstrBtn,
            self.similarityScoreBtn
        ):
            btn.setMinimumWidth(200)

        self.substrFilterStatusLabel = QLabel("")
        self.substrFilterStatusLabel.setVisible(False)

        self.substrFilterProgressBar = QProgressBar()
        self.substrFilterProgressBar.setRange(0, 100)
        self.substrFilterProgressBar.setValue(0)
        self.substrFilterProgressBar.setVisible(False)

        layout = QVBoxLayout()

        for btn in (
            self.filterLibraryBtn,
            self.filterSubstrBtn,
            self.similarityScoreBtn
        ):
            layout.addWidget(btn)

        layout.addWidget(self.substrFilterStatusLabel)
        layout.addWidget(self.substrFilterProgressBar)

        self.filtrationGroupBox.setLayout(layout)

    def createManageLibraryGroupBox(self):
        self.manageLibraryGroupBox = QGroupBox("Manage Library")

        self.splitDatabaseBtn = QPushButton("Split the Database")
        self.splitDatabaseBtn.clicked.connect(self.splitDatabase)
        self.batchSizeSpinBox = QSpinBox()
        self.batchSizeSpinBox.setRange(1, 1000000)
        self.batchSizeSpinBox.setValue(3500)
        self.mergeDatabaseBtn = QPushButton("Merge Databases")
        self.mergeDatabaseBtn.clicked.connect(self.mergeDatabase)

        batch_row = QHBoxLayout()
        batch_row.addWidget(QLabel("Batch size:"))
        batch_row.addWidget(self.batchSizeSpinBox)

        layout = QVBoxLayout()
        layout.addWidget(self.splitDatabaseBtn)
        layout.addLayout(batch_row)
        layout.addWidget(self.mergeDatabaseBtn)
        self.manageLibraryGroupBox.setLayout(layout)

    def on_filter_substr_clicked(self):
        dialog = FilterByMoietyDialog(
            self,
            checked_labels=self.active_filter_labels,
            match_mode=getattr(self, "active_match_mode", "all"),
            exclude_pains=getattr(self, "active_exclude_pains", False),
            exclude_brenk=getattr(self, "active_exclude_brenk", False),
            exclude_nih=getattr(self, "active_exclude_nih", False),
            exclude_chembl=getattr(self, "active_exclude_chembl", False),
        )
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.active_filter_labels = set(dialog.selected_labels)
            self.active_match_mode = dialog.match_mode
            self.active_exclude_pains = dialog.exclude_pains
            self.active_exclude_brenk = dialog.exclude_brenk
            self.active_exclude_nih = dialog.exclude_nih
            self.active_exclude_chembl = dialog.exclude_chembl
            self.apply_filters(dialog.selected_filters, dialog.match_mode, dialog.exclude_pains,
                                dialog.exclude_brenk, dialog.exclude_nih, dialog.exclude_chembl)

    def apply_filters(self, smarts_list, match_mode="all",
                      exclude_pains=False, exclude_brenk=False,
                      exclude_nih=False, exclude_chembl=False
    ):

        if self._substr_thread is not None:
            try:
                if self._substr_thread.isRunning():
                    QMessageBox.information(
                        self,
                        "Filtration running",
                        "A substructure filtration is already in progress."
                    )
                    return
            except RuntimeError:
                self._substr_thread = None
                self._substr_worker = None

        self.filterLibraryBtn.setDisabled(True)
        self.filterSubstrBtn.setDisabled(True)
        self.similarityScoreBtn.setDisabled(True)

        self.selectReactionGroupBox.setDisabled(True)
        self.runGroupBox.setDisabled(True)
        self.manageLibraryGroupBox.setDisabled(True)
        self.exportGroupBox.setDisabled(True)

        self.substrFilterStatusLabel.setText(
            f"Filtering {len(baseSMILES):,} molecule(s) by substructure…"
        )
        self.substrFilterStatusLabel.setVisible(True)

        self.substrFilterProgressBar.setValue(0)
        self.substrFilterProgressBar.setVisible(True)

        selected = sorted(self.active_filter_labels)
        filter_parts = []

        if selected:
            mode = "ALL" if match_mode == "all" else "ANY"
            filter_parts.append(
                f"{mode} selected substructures: {', '.join(selected)}"
            )

        if exclude_pains:
            filter_parts.append("exclude PAINS")
        if exclude_brenk:
            filter_parts.append("exclude Brenk alerts")
        if exclude_nih:
            filter_parts.append("exclude NIH alerts")
        if exclude_chembl:
            filter_parts.append("exclude ChEMBL alerts")

        description = "; ".join(filter_parts) if filter_parts else "no filters selected"

        self.log_message(
            f"Substructure filtration started: "
            f"{len(baseSMILES):,} molecule(s); "
            f"{description}."
        )

        smiles_snapshot = list(baseSMILES)
        self._substr_thread = QThread()
        self._substr_worker = SubstructureFilterWorker(
            smiles_snapshot,
            smarts_list,
            match_mode,
            exclude_pains,
            exclude_brenk,
            exclude_nih,
            exclude_chembl
        )

        self._substr_worker.moveToThread(self._substr_thread)
        self._substr_thread.started.connect(
            self._substr_worker.run
        )
        self._substr_worker.progress.connect(self._on_substr_filter_progress)
        self._substr_worker.finished.connect(self._on_substr_filter_finished)
        self._substr_worker.error.connect(self._on_substr_filter_error)
        self._substr_worker.finished.connect(self._substr_thread.quit)
        self._substr_worker.error.connect(self._substr_thread.quit)
        self._substr_worker.finished.connect(self._substr_worker.deleteLater)
        self._substr_thread.finished.connect(self._substr_thread.deleteLater)
        self._substr_thread.finished.connect(lambda: setattr(self, "_substr_thread", None))
        self._substr_thread.finished.connect(lambda: setattr(self, "_substr_worker", None))
        self._substr_thread.start()

    def _on_substr_filter_progress(self, value):
        self.substrFilterProgressBar.setValue(value)

        self.substrFilterStatusLabel.setText(
            f"Filtering library by substructure… {value}%"
        )
    
    def _on_substr_filter_finished(self, result):
        global baseSMILES

        baseSMILES = result["filtered_smiles"]
        self.library_df = None

        self.substrFilterProgressBar.setValue(100)

        self.substrFilterProgressBar.setVisible(False)
        self.substrFilterStatusLabel.setVisible(False)
        self.substrFilterStatusLabel.setText("")

        self.log_message(
            f"Substructure filtration complete: "
            f"{result['before']:,} → {result['after']:,} molecule(s)."
        )

        self.refreshFiltrationAvailability()

        self.manageLibraryGroupBox.setDisabled(False)

        if len(baseSMILES) > 0:
            self.selectReactionGroupBox.setDisabled(False)

        self.filterLibraryBtn.setDisabled(False)
        self.filterSubstrBtn.setDisabled(False)
        self.similarityScoreBtn.setDisabled(False)

    def _on_substr_filter_error(self, message):
        self.substrFilterProgressBar.setVisible(False)
        self.substrFilterStatusLabel.setText("Substructure filtration failed.")
        self.substrFilterStatusLabel.setVisible(True)
        self.filterLibraryBtn.setDisabled(False)
        self.filterSubstrBtn.setDisabled(False)
        self.similarityScoreBtn.setDisabled(False)

        self.manageLibraryGroupBox.setDisabled(False)

        if len(baseSMILES) > 0:
            self.selectReactionGroupBox.setDisabled(False)

        self.refreshFiltrationAvailability()

        QMessageBox.critical(
            self,
            "Filter failed",
            message
        )

    def on_filter_clicked(self):
        global baseSMILES

        if (
            getattr(self, "library_df", None) is not None
            and not self.library_df.empty
        ):
            self.open_filtration_window()
            return

        if not baseSMILES:
            QMessageBox.information(
                self,
                "No library",
                "Load or generate a library first."
            )
            return

        if self._descriptor_thread is not None:
            try:
                if self._descriptor_thread.isRunning():
                    QMessageBox.information(
                        self,
                        "Descriptor calculation running",
                        "Molecular descriptors are already being calculated."
                    )
                    return
            except RuntimeError:
                self._descriptor_thread = None
                self._descriptor_worker = None

        self.filterLibraryBtn.setDisabled(True)
        self.filterSubstrBtn.setDisabled(True)
        self.similarityScoreBtn.setDisabled(True)

        self.selectReactionGroupBox.setDisabled(True)
        self.runGroupBox.setDisabled(True)
        self.manageLibraryGroupBox.setDisabled(True)
        self.exportGroupBox.setDisabled(True)

        self.substrFilterStatusLabel.setText(
            f"Calculating molecular descriptors for "
            f"{len(baseSMILES):,} molecule(s)…"
        )
        self.substrFilterStatusLabel.setVisible(True)
        self.substrFilterProgressBar.setRange(0, 100)
        self.substrFilterProgressBar.setValue(0)
        self.substrFilterProgressBar.setVisible(True)

        smiles_snapshot = list(baseSMILES)

        self._descriptor_thread = QThread()
        self._descriptor_worker = DescriptorWorker(
            smiles_list=smiles_snapshot,
            batch_size=2000
        )
        self._descriptor_worker.moveToThread(self._descriptor_thread)
        self._descriptor_thread.started.connect(self._descriptor_worker.run)
        self._descriptor_worker.progress.connect(self._on_descriptor_progress)
        self._descriptor_worker.finished.connect(self._on_descriptor_finished)
        self._descriptor_worker.error.connect(self._on_descriptor_error)
        self._descriptor_worker.finished.connect(self._descriptor_thread.quit)
        self._descriptor_worker.error.connect(self._descriptor_thread.quit)
        self._descriptor_worker.finished.connect(self._descriptor_worker.deleteLater)
        self._descriptor_worker.error.connect(self._descriptor_worker.deleteLater)
        self._descriptor_thread.finished.connect(self._descriptor_thread.deleteLater)

        self._descriptor_thread.finished.connect(lambda: setattr(self,"_descriptor_thread",None))
        self._descriptor_thread.finished.connect(lambda: setattr( self, "_descriptor_worker", None))

        self._descriptor_thread.start()

    def _on_descriptor_progress(self, value):
        self.substrFilterProgressBar.setValue(value)

        self.substrFilterStatusLabel.setText(
            f"Calculating molecular descriptors… {value}%"
        )


    def _on_descriptor_finished(self, df):
        self.library_df = df
        self.substrFilterProgressBar.setVisible(False)
        self.substrFilterStatusLabel.setVisible(False)
        self.substrFilterStatusLabel.setText("")
        self.filterLibraryBtn.setDisabled(False)
        self.filterSubstrBtn.setDisabled(False)
        self.similarityScoreBtn.setDisabled(False)
        self.manageLibraryGroupBox.setDisabled(False)
        self.exportGroupBox.setDisabled(False)

        if len(baseSMILES) > 0:
            self.selectReactionGroupBox.setDisabled(False)

        self.refreshFiltrationAvailability()

        self.log_message(
            f"Descriptors calculated: "
            f"{len(df):,} molecule(s)."
        )

        self.open_filtration_window()


    def _on_descriptor_error(self, message):
        self.substrFilterProgressBar.setVisible(False)
        self.substrFilterStatusLabel.setText("Descriptor calculation failed.")
        self.substrFilterStatusLabel.setVisible(True)
        self.filterLibraryBtn.setDisabled(False)
        self.filterSubstrBtn.setDisabled(False)
        self.similarityScoreBtn.setDisabled(False)
        self.manageLibraryGroupBox.setDisabled(False)
        self.exportGroupBox.setDisabled(False)

        if len(baseSMILES) > 0:
            self.selectReactionGroupBox.setDisabled(False)

        self.refreshFiltrationAvailability()

        QMessageBox.critical(self, "Descriptor calculation failed", message)

    def open_filtration_window(self):
        self.filterLibraryBtn.setDisabled(True)
        if getattr(self, "library_df", None) is None or self.library_df.empty:
            QMessageBox.information(self, "No library", "Load or build a library first.")
            return
        if self.second is None or not self.second.isVisible():
            self.second = FiltrationWindow(parent=None, df=self.library_df)
            self.second.setWindowFlag(Qt.WindowType.Window, True)
            self.second.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
            self.second.destroyed.connect(lambda: setattr(self, "second", None))

        geo = self.frameGeometry()
        self.second.move(geo.topRight() + QPoint(20, 0))

        self.second.showNormal()
        self.second.raise_()
        self.second.activateWindow()

    def open_similarity_window(self):
        if not baseSMILES:
            QMessageBox.information(self, "No library", "Load or build a library first.")
            return
        if getattr(self, "library_df", None) is None or self.library_df.empty:
            try:
                self.library_df = self.build_library_df()
            except Exception as e:
                QMessageBox.critical(self, "Build failed", f"{type(e).__name__}: {e}")
                return
        if self.third is None or not self.third.isVisible():
            self.third = SimilarityWindow(parent=None, df=self.library_df)
            self.third.setWindowFlag(Qt.WindowType.Window, True)
            self.third.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
            self.third.destroyed.connect(lambda: setattr(self, "third", None))

        geo = self.frameGeometry()
        self.third.move(geo.topRight() + QPoint(20, 240))

        self.third.showNormal()
        self.third.raise_()
        self.third.activateWindow()

    def activateNameBtn(self):
        if self.libraryNameBox.text() != "":
            self.libraryNameBtn.setDisabled(False)
        else:
            self.libraryNameBtn.setDisabled(True)

    def startRxnQThread(self):
        global reaction
        if reaction == "Ugi Reaction": 
            self.thread = UgiReaction()
        else:
            self.thread = RxnQThread()
        self.thread.finished.connect(self.afterReaction)
        self.thread.finished.connect(self.thread.quit)
        self.thread.start()

    def setName(self):
        global name
        name = str(self.libraryNameBox.text())
        self.setWindowTitle("MOLDE: "+name)
        self.importMoleculesGroupBox.setDisabled(False)
        self.libraryNameBox.setDisabled(True)
        self.libraryNameBtn.setDisabled(True)
        self.setReactionBtn.setDisabled(True)

    def createImportMoleculesGroupBox(self):
        self.importMoleculesGroupBox = QGroupBox("Import Molecules")
        self.importBaseBtn = QPushButton('Import Starting Materials')
        self.importBaseBtn.clicked.connect(self.importBases)
        self.loopProducts = QCheckBox("Using Products\nfrom Last Step")
        self.loopProducts.setChecked(False)
        self.loopProducts.setDisabled(True)
        self.importReactantsBtn = QPushButton('Import Reactants')
        self.importReactantsBtn.clicked.connect(self.importReactants)
        self.importReactantsBtn.setDisabled(True)
        layout = QVBoxLayout()
        layout.addWidget(self.importBaseBtn)
        layout.addWidget(self.loopProducts)
        layout.addWidget(self.importReactantsBtn)
        self.importMoleculesGroupBox.setLayout(layout)

    def createSelectReactionGroupBox(self):
        global oneReactions, twoReactions
        self.selectReactionGroupBox = QGroupBox("Select Reaction")

        self.twoComboBox = QComboBox()
        self.twoComboBox.addItems(twoReactions)
        self.twoComboBox.textActivated.connect(self.twoReactionSet)
        self.oneComboBox = QComboBox()
        self.oneComboBox.addItems(oneReactions)
        self.oneComboBox.textActivated.connect(self.oneReactionSet)
        self.abbaComboBox = QComboBox()
        self.abbaComboBox.addItems(["Select Pairing Mode", "Starting Material is A", "Starting Material is B", "Reactant Pool"])
        self.abbaComboBox.currentTextChanged.connect(self.abbaModeChanged)
        self.AB_textbox = QLabel("\n")
        self.retainStartingCheckBox = QCheckBox("Retain Starting Reactants?")
        self.setReactionBtn = QPushButton("Set Reaction Choice")
        self.setReactionBtn.clicked.connect(self.setReaction)

        layout = QGridLayout()
        layout.addWidget(self.twoComboBox, 0, 0)
        layout.addWidget(self.oneComboBox, 1, 0)
        layout.addWidget(self.abbaComboBox, 2, 0)
        layout.addWidget(self.AB_textbox, 3, 0)
        layout.addWidget(self.retainStartingCheckBox, 4, 0)
        layout.addWidget(self.setReactionBtn, 5, 0)
        self.selectReactionGroupBox.setLayout(layout)

    def abbaModeChanged(self, text):
        if not reaction or reaction not in reactions_dictionary:
            return
        if reactions_dictionary[reaction]["type"] == "multi":
            return
        if text == "Reactant Pool":
            self.AB_textbox.setText("Reactant Pool Mode\nNo A/B Roles Assigned")
        elif reactions_dictionary[reaction]["type"] == "two":
            self.AB_textbox.setText("A: "+reactions_dictionary[reaction]["A"]+"\nB: "+reactions_dictionary[reaction]["B"])

    def createMenuBar(self):
        self.menuBar = QMenuBar(self)
        programMenu = self.menuBar.addMenu("Program")
        programMenu.addAction("Reset Selections", self.resetSelection)
        programMenu.addAction("Reset All", self.resetAll)
        programMenu.addAction("Set Working Directory", self.chooseWorkingDirectory)

    def createRunGroupBox(self):
        self.runGroupBox = QGroupBox("Run Reaction")

        execBtn = QPushButton('Run Reaction')
        execBtn.clicked.connect(self.executeReaction)
        execBtn.setMinimumHeight(80)
        runFont = execBtn.font()
        runFont.setPointSize(runFont.pointSize() + 4)
        runFont.setBold(True)
        execBtn.setFont(runFont)

        self.reactionProgressBar = QProgressBar()
        self.reactionProgressBar.setRange(0, 0)
        self.reactionProgressBar.setVisible(False)

        layout = QVBoxLayout()
        layout.addWidget(execBtn)
        layout.addWidget(self.reactionProgressBar)
        self.runGroupBox.setLayout(layout)

    def createVisualizationGroupBox(self):
        self.visualizationGroupBox = QGroupBox("Visualization")
        self.visualizeInitialBtn = QPushButton('Visualize Initial\nDatabase')
        self.visualizeInitialBtn.clicked.connect(self.showInitialBaseImageViewer)
        self.visualizeReactantsBtn = QPushButton('Visualize\nReactants')
        self.visualizeReactantsBtn.clicked.connect(self.showReactantImageViewer)
        self.visualizeCurrentBtn = QPushButton('Current\nDatabase')
        self.visualizeCurrentBtn.clicked.connect(self.showBaseImageViewer)
        layout = QVBoxLayout()
        layout.addWidget(self.visualizeInitialBtn)
        layout.addWidget(self.visualizeReactantsBtn)
        layout.addWidget(self.visualizeCurrentBtn)
        self.visualizationGroupBox.setLayout(layout)

    def _openImageViewer(self, smiles_list):
        global viewerList
        global baseNum

        if getattr(self, "w", None) is not None:
            self.w.close()

        baseNum = 0
        viewerList = smiles_list
        self.w = imageViewer()
        self.w.show()

    def showInitialBaseImageViewer(self):
        global initialBaseSMILES
        self._openImageViewer(initialBaseSMILES)

    def showBaseImageViewer(self):
        global baseSMILES
        self._openImageViewer(baseSMILES)

    def showReactantImageViewer(self):
        global lastReactantSMILES
        self._openImageViewer(lastReactantSMILES)

    def log_message(self, message):
        self.textbox.append(message)

    def createMiddleGroupBox(self):
        self.middleGroupBox = QGroupBox("Log")
        self.textbox = QTextEdit()
        self.textbox.setReadOnly(True)
        if startup_log:
            self.textbox.setPlainText(startup_log.strip())
        else:
            self.textbox.clear()
        layout = QVBoxLayout()
        layout.addWidget(self.textbox)

        self.middleGroupBox.setLayout(layout)

    def createExportGroupBox(self):
        self.exportGroupBox = QGroupBox("Export Final Products")

        self.ExportDirBtn = QPushButton('Choose Export Directory')
        self.ExportDirBtn.clicked.connect(self.setExportPath)

        self.fileTypeComboBox = QComboBox()
        self.fileTypeComboBox.addItems([
            ".sdf (Molecules only)",
            ".sdf (Molecules + Properties)",
            ".mol (Individual files)",
            "SMILES as .csv"
        ])

        self.finalExportButton = QPushButton("Export Products")
        self.finalExportButton.clicked.connect(self.exportProducts)

        self.fileTypeComboBox.setDisabled(True)
        self.finalExportButton.setDisabled(True)

        self.exportDirLabel = QLabel("Selected Directory:")

        self.exportProgressBar = QProgressBar()
        self.exportProgressBar.setRange(0, 100)
        self.exportProgressBar.setVisible(False)

        dir_row = QHBoxLayout()
        dir_row.addWidget(self.ExportDirBtn)
        dir_row.addWidget(self.fileTypeComboBox)

        layout = QVBoxLayout()
        layout.addWidget(self.exportDirLabel)
        layout.addLayout(dir_row)
        layout.addWidget(self.finalExportButton)
        layout.addWidget(self.exportProgressBar)

        self.exportGroupBox.setLayout(layout)

    def refreshFiltrationAvailability(self):
        has_library = bool(baseSMILES)
        self.filterLibraryBtn.setEnabled(has_library)
        self.similarityScoreBtn.setEnabled(has_library)
        self.selectReactionGroupBox.setEnabled(has_library)
        self.mergeDatabaseBtn.setEnabled(has_library)
        self.splitDatabaseBtn.setEnabled(has_library)
        self.exportGroupBox.setEnabled(has_library)
        self.visualizationGroupBox.setEnabled(has_library)
        self.filterSubstrBtn.setEnabled(has_library)

    def setReaction(self):
        global abbaVar

        abbaVar = str(self.abbaComboBox.currentText())

        if reactions_dictionary[reaction]["type"] == "multi":
            self.log_message(
                f"Reaction selected: {reaction} "
                f"(multi-component, reactant pool mode)."
            )
            self.importReactantsBtn.setDisabled(False)
            self.runGroupBox.setDisabled(False)

        elif reactions_dictionary[reaction]["type"] == "two":
            if abbaVar == "Reactant Pool":
                self.log_message(
                    f"Reaction selected: {reaction} "
                    f"(two-component, reactant pool mode)."
                )
                self.runGroupBox.setDisabled(False)
            else:
                self.log_message(
                    f"Reaction selected: {reaction} "
                    f"(two-component, {abbaVar})."
                )

            self.importReactantsBtn.setDisabled(False)

        elif reactions_dictionary[reaction]["type"] == "one":
            self.log_message(
                f"Reaction selected: {reaction} (one-component)."
            )
            self.runGroupBox.setDisabled(False)

        else:
            print("Error")

    def oneReactionSet(self, opt):
        global reaction
        global reactions_dictionary

        if opt != "One Component":
            self.twoComboBox.setDisabled(True)
            self.abbaComboBox.setDisabled(True)
            self.setReactionBtn.setDisabled(False)
            self.retainStartingCheckBox.setDisabled(False)
            self.retainStartingCheckBox.setCheckState(Qt.CheckState.Checked)
            self.AB_textbox.setText("Starting Material: "+reactions_dictionary[opt]["Starting Material"]+"\nProduct: "+reactions_dictionary[opt]["Product"])
            reaction = opt

        else:
            self.AB_textbox.setText("\n")
            self.twoComboBox.setDisabled(False)
            self.abbaComboBox.setDisabled(False)
            self.setReactionBtn.setDisabled(True)
            self.retainStartingCheckBox.setDisabled(True)
            self.retainStartingCheckBox.setCheckState(Qt.CheckState.Unchecked)

    def twoReactionSet(self, opt):
        global reaction
        global reactions_dictionary

        if opt != "Multi Component":
            self.oneComboBox.setDisabled(True)
            self.setReactionBtn.setDisabled(False)
            self.retainStartingCheckBox.setDisabled(False)
            self.retainStartingCheckBox.setCheckState(Qt.CheckState.Unchecked)
            reaction = opt
            if reactions_dictionary[reaction]["type"] == "multi":
                self.AB_textbox.setText("A/B Reactant Roles\nAre Not Supported\nFor Reaction")
                self.abbaComboBox.setCurrentIndex(self.abbaComboBox.findText("Reactant Pool"))
                self.abbaComboBox.setDisabled(True)
            else:
                self.abbaModeChanged(self.abbaComboBox.currentText())
                self.abbaComboBox.setDisabled(False)
        else:
            self.AB_textbox.setText("\n")
            self.oneComboBox.setDisabled(False)
            self.setReactionBtn.setDisabled(True)
            self.retainStartingCheckBox.setDisabled(True)
            self.retainStartingCheckBox.setCheckState(Qt.CheckState.Unchecked)

    def importBases(self):
        global baseSMILES
        global initialBaseSMILES

        files = self.getFiles()

        if not files:
            return

        total_before = len(baseSMILES)

        for path in files:
            imported_smiles = self.fileImport(path)
            baseSMILES.extend(imported_smiles)

            self.log_message(
                f"Starting materials imported: "
                f"{len(imported_smiles):,} molecule(s) from "
                f"{os.path.basename(path)}."
            )

        baseSMILES = DeDuplicate(baseSMILES)
        added = len(baseSMILES) - total_before

        self.log_message(
            f"Starting-material library: {len(baseSMILES):,} unique molecule(s) "
            f"({added:,} added)."
        )

        if len(baseSMILES) > 0:
            initialBaseSMILES = list(baseSMILES)
            self.importBaseBtn.setDisabled(True)
            self.library_df = None
            self.refreshFiltrationAvailability()

    def importReactants(self):
        global reactantSMILES
        global lastReactantSMILES

        reactantSMILES = []

        files = self.getFiles()

        if not files:
            return

        for path in files:
            imported_smiles = self.fileImport(path)
            reactantSMILES.extend(imported_smiles)

            self.log_message(
                f"Reactants imported: "
                f"{len(imported_smiles):,} molecule(s) from "
                f"{os.path.basename(path)}."
            )

        reactantSMILES = DeDuplicate(reactantSMILES)

        self.log_message(
            f"Reactant library: {len(reactantSMILES):,} unique molecule(s)."
        )

        if len(reactantSMILES) > 0:
            lastReactantSMILES = list(reactantSMILES)
            self.runGroupBox.setDisabled(False)
            self.refreshFiltrationAvailability()

    def getFiles(self):
        global workingDirectory

        file_list = []
        file_filter = 'SD File (*.sdf)'
        response = QFileDialog.getOpenFileNames(
            parent=self,
            caption='Select file(s)',
            directory=workingDirectory,
            filter=file_filter,
            initialFilter='SD File (*.sdf)'
        )

        if response[0]:
            workingDirectory = os.path.dirname(response[0][0])
        x = 0
        for i in response[:-1]: 
            for l in i: 
                file_list.append(l)
            x = x+1
        return file_list

    def mergeDatabase(self):
        global baseSMILES

        files = self.getFiles()

        if not files:
            return

        before = len(baseSMILES)
        imported_total = 0
        merged_names = []

        for path in files:
            merged_smiles = self.fileImport(path)
            imported_total += len(merged_smiles)
            baseSMILES.extend(merged_smiles)
            merged_names.append(os.path.basename(path))

        baseSMILES = DeDuplicate(baseSMILES)
        added = len(baseSMILES) - before
        names = ", ".join(merged_names)

        self.log_message(
            f'Library merge: current library + {names}. '
            f'{imported_total:,} molecule(s) imported; '
            f'{added:,} new unique molecule(s) added; '
            f'library now contains {len(baseSMILES):,} molecule(s).'
        )

        self.library_df = None
        self.refreshFiltrationAvailability()

    def splitDatabase(self):
        global baseSMILES, exportPath, name

        if not exportPath or not os.path.isdir(exportPath):
            QMessageBox.warning(self, "No directory", "Choose an export directory first.")
            return

        descriptor_lookup = None

        if getattr(self, "library_df", None) is not None and not self.library_df.empty and "SMILES" in self.library_df.columns:
            descriptor_lookup = _ensure_descriptor_columns(self.library_df.copy()).drop_duplicates(subset="SMILES", keep="first").set_index("SMILES")

        self.splitDatabaseBtn.setDisabled(True)
        self.exportProgressBar.setVisible(True)
        self._split_thread = QThread()
        self._split_worker = SplitWorker(
            list(baseSMILES), exportPath, name,
            batch_size=self.batchSizeSpinBox.value(),
            descriptor_lookup=descriptor_lookup,
        )
        self._split_worker.moveToThread(self._split_thread)
        self._split_thread.started.connect(self._split_worker.run)
        self._split_worker.progress.connect(self.exportProgressBar.setValue)
        self._split_worker.finished.connect(self._on_split_finished)
        self._split_worker.error.connect(self._on_split_error)
        self._split_worker.finished.connect(self._split_thread.quit)
        self._split_worker.error.connect(self._split_thread.quit)
        self._split_thread.start()

    def _on_split_finished(self, result):
        self.splitDatabaseBtn.setDisabled(False)
        self.exportProgressBar.setVisible(False)

        message = (
            f"Library split: {result['total_molecules']:,} molecule(s) "
            f"written to {result['file_count']:,} file(s)"
        )

        if result["skipped"]:
            message += (
                f"; {result['skipped']:,} invalid molecule(s) skipped"
            )

        self.log_message(message + ".")

    def _on_split_error(self, message):
        self.splitDatabaseBtn.setDisabled(False)
        self.exportProgressBar.setVisible(False)
        QMessageBox.critical(self, "Split failed", message)

    def setExportPath(self):
        global exportPath
        global workingDirectory

        if exportPath == "":
            exportPath = QFileDialog.getExistingDirectory(
                directory=workingDirectory,
                caption='Select the export directory.',
                options=QFileDialog.Option.DontUseNativeDialog
                )
            if exportPath:
                workingDirectory = exportPath
            self.exportDirLabel.setText("Selected Directory:\n"+exportPath+"/")
            self.ExportDirBtn.setText("Reset Directory")
            self.fileTypeComboBox.setDisabled(False)
            self.finalExportButton.setDisabled(False)
            self.splitDatabaseBtn.setDisabled(False)
        else:
            exportPath = ""
            self.exportDirLabel.setText("Selected Directory: ")
            self.ExportDirBtn.setText("Choose Export Directory")
            self.fileTypeComboBox.setDisabled(True)
            self.finalExportButton.setDisabled(True)
            self.splitDatabaseBtn.setDisabled(True)

    def chooseWorkingDirectory(self):
        global workingDirectory

        chosen = QFileDialog.getExistingDirectory(parent=self, caption='Choose your working directory', directory=workingDirectory,)

        if chosen:
            workingDirectory = chosen

    def fileImport(self, path_, enumerate_stereoisomers=False):
        smis = []
        threads = max(1, int(os.cpu_count())-2) 
        
        with Chem.MultithreadedSDMolSupplier(path_, numWriterThreads=threads) as suppl:
            for mol in suppl:
                if mol is None: 
                    continue   
                smis.append(Chem.MolToSmiles(mol))

        return smis

    def resetReactionSelectionWidgets(self):
        self.oneComboBox.setCurrentIndex(0)
        self.twoComboBox.setCurrentIndex(0)
        self.abbaComboBox.setCurrentIndex(0)
        self.oneComboBox.setDisabled(False)
        self.twoComboBox.setDisabled(False)
        self.abbaComboBox.setDisabled(False)
        self.setReactionBtn.setDisabled(True)
        self.retainStartingCheckBox.setDisabled(True)
        self.retainStartingCheckBox.setCheckState(Qt.CheckState.Unchecked)
        self.AB_textbox.setText("\n")

    def resetSelection(self, clear_log=False):
        global reaction
        global baseFiles
        global baseSMILES
        global reactantSMILES
        global smarts
        global smis_a
        global smis_b
        global baseNum

        reaction = ""
        reactantSMILES = []
        smarts = ""
        smis_a = []
        smis_b = []
        baseNum = 0

        self.resetReactionSelectionWidgets()

        self.importReactantsBtn.setDisabled(True)
        self.runGroupBox.setDisabled(True)

        self.refreshFiltrationAvailability()

        self.importBaseBtn.setDisabled(
            False if not baseSMILES else True
        )
    
        if getattr(self, "second", None) is not None:
            self.second.close()
            self.second = None

    def resetAll(self):
        global exportPath
        global name
        global baseSMILES
        global initialBaseSMILES
        global reactantSMILES
        global lastReactantSMILES
        global baseFiles
        global reaction
        global rxn_products
        global smarts
        global smis_a
        global smis_b
        global baseNum

        exportPath = ""
        name = ""
        baseSMILES = []
        initialBaseSMILES = []
        reactantSMILES = []
        lastReactantSMILES = []
        baseFiles = []
        reaction = ""
        rxn_products = []
        smarts = ""
        smis_a = []
        smis_b = []
        baseNum = 0

        self.library_df = None
        self.textbox.setText("")
        self.libraryNameBox.setText("")
        self.resetReactionSelectionWidgets()
        self.loopProducts.setChecked(False)
        self.loopProducts.setDisabled(True)
        self.libraryNameBox.setDisabled(False)
        self.libraryNameBtn.setDisabled(True)
        self.importMoleculesGroupBox.setDisabled(True)
        self.importBaseBtn.setDisabled(False)
        self.importReactantsBtn.setDisabled(True)
        self.selectReactionGroupBox.setDisabled(True)
        self.runGroupBox.setDisabled(True)
        self.exportGroupBox.setDisabled(True)
        self.exportDirLabel.setText("Selected Directory: ")
        self.ExportDirBtn.setText("Choose Export Directory")
        self.fileTypeComboBox.setDisabled(True)
        self.mergeDatabaseBtn.setDisabled(True)
        self.splitDatabaseBtn.setDisabled(True)
        self.finalExportButton.setDisabled(True)
        self.visualizationGroupBox.setDisabled(True)
        self.reactionHasRun = False
        self.filterLibraryBtn.setDisabled(True)
        self.similarityScoreBtn.setDisabled(True)
        self.filterSubstrBtn.setDisabled(True)

        if getattr(self, "second", None) is not None:
            self.second.close()
            self.second = None
        if getattr(self, "third", None) is not None:
            self.third.scored_df = None
            self.third.filtered_df = None
            self.third.close()
            self.third = None

    def executeReaction(self):
        global smarts
        global reactions_dictionary
        global smis_a
        global smis_b
        global reactant_pool
        global rxn_products
        global baseSMILES
        global reactantSMILES
        global start
        global finish
        global reaction
        global abbaVar
        self.runGroupBox.setDisabled(True)
        self.reactionProgressBar.setVisible(True)
        self.log_message(
            f"Reaction started: {reaction} on "
            f"{len(baseSMILES):,} starting molecule(s)."
        )

        self.importReactantsBtn.setDisabled(True)
        smis_a = []
        smis_b = []
        rxn_products = []

        if self.retainStartingCheckBox.isChecked() == True:
            rxn_products.extend(baseSMILES)
            rxn_products.extend(reactantSMILES)

        type_ = reactions_dictionary[reaction]["type"]

        if type_ == "two" or type_ == "multi":

            if abbaVar == "Starting Material is A":
                smis_a.extend(baseSMILES)
                smis_b.extend(reactantSMILES)
            elif abbaVar == "Starting Material is B":
                smis_a.extend(reactantSMILES)
                smis_b.extend(baseSMILES)
            elif abbaVar == "Reactant Pool":
                reactant_pool.extend(baseSMILES)
                reactant_pool.extend(reactantSMILES)
            if type_ == "two":
                smarts = genSMARTS(reaction)

            start = time.perf_counter()
            self.startRxnQThread()

        elif type_ == "one":
            smis_a.extend(baseSMILES)
            start = time.perf_counter()
            self.startRxnQThread()

    def afterReaction(self):
        self.importReactantsBtn.setDisabled(False)
        global smarts
        global smis_b
        global rxn_products
        global baseSMILES
        global reactantSMILES
        global start
        global finish
        self.reactionHasRun = True
        self.reactionProgressBar.setVisible(False)
        starting_size = len(baseSMILES)
        baseSMILES = []

        baseSMILES.extend(DeDuplicate(rxn_products))

        self.library_df = None
        finish=time.perf_counter()
        final_size = len(baseSMILES)

        if final_size > starting_size:
            change_note = (
                f"{final_size - starting_size:,} more than the "
                f"{starting_size:,} starting molecules"
            )
        elif final_size < starting_size:
            change_note = (
                f"{starting_size - final_size:,} fewer than the "
                f"{starting_size:,} starting molecules; "
                f"some had no valid reaction site or products were "
                f"removed during de-duplication"
            )
        else:
            change_note = (
                f"same count as the {starting_size:,} starting molecules"
            )
        self.log_message(
            f"Reaction complete: {final_size:,} unique molecule(s) generated "
            f"({change_note}); completed in {finish - start:.2f} s."
        )
        
        self.resetSelection()
        self.loopProducts.setChecked(True)
        self.importBaseBtn.setDisabled(True)
        self.refreshFiltrationAvailability()
        rxn_products = []

    def exportProducts(self):
        global exportPath
        global name

        if not exportPath or not os.path.isdir(exportPath):
            QMessageBox.critical(
                self,
                "No export directory",
                "Please choose (or re-choose) an export directory before exporting."
            )
            return

        if self._exportThread is not None and self._exportThread.isRunning():
            QMessageBox.information(
                self,
                "Export running",
                "An export is already in progress."
            )
            return

        export_type = str(self.fileTypeComboBox.currentText())

        include_properties = export_type in (".sdf (Molecules + Properties)", "SMILES as .csv",)

        descriptor_lookup = None

        if include_properties:
            if (getattr(self, "library_df", None) is not None and not self.library_df.empty and "SMILES" in self.library_df.columns):
                descriptor_df = _ensure_descriptor_columns(self.library_df.copy())
            else:
                descriptor_df = _ensure_descriptor_columns(pd.DataFrame({"SMILES": baseSMILES}))

            descriptor_lookup = (descriptor_df.drop_duplicates(subset="SMILES", keep="first").set_index("SMILES"))

        self.log_message(
            f"Export started: {len(baseSMILES):,} molecule(s), "
            f"{export_type}."
        )

        self.finalExportButton.setDisabled(True)
        self.exportProgressBar.setVisible(True)
        self.exportProgressBar.setValue(0)

        self._exportThread = QThread()

        self._exportWorker = ExportWorker(
            smiles_list=list(baseSMILES),
            descriptor_lookup=descriptor_lookup,
            export_path=exportPath,
            file_name=name,
            file_type=export_type,
            include_properties=include_properties
        )

        self._exportWorker.moveToThread(self._exportThread)

        self._exportThread.started.connect(self._exportWorker.run)
        self._exportWorker.progress.connect(self.exportProgressBar.setValue)

        self._exportWorker.finished.connect(self._on_export_finished)
        self._exportWorker.error.connect(self._on_export_error)

        self._exportWorker.error.connect(self._exportThread.quit)
        self._exportWorker.finished.connect(self._exportThread.quit)

        self._exportWorker.finished.connect(self._exportWorker.deleteLater)
        self._exportThread.finished.connect(self._exportThread.deleteLater)

        self._exportThread.finished.connect(lambda: setattr(self, "_exportThread", None))
        self._exportWorker.finished.connect(lambda: setattr(self, "_exportWorker", None))

        self._exportThread.start()

    def _on_export_error(self, message: str):
        self.finalExportButton.setDisabled(False)
        self.exportProgressBar.setVisible(False)
        QMessageBox.critical(self, "Export failed", message)
        print(message)

    def _on_export_finished(self, payload: dict):
        self.finalExportButton.setDisabled(False)
        self.exportProgressBar.setVisible(False)
        exported = payload.get("exported", 0)
        skipped = payload.get("skipped", 0)
        skipped_smiles = payload.get("skipped_smiles", [])
        self.log_message(
            f"Export complete: {exported:,} molecule(s) written."
        )

        if skipped:
            self.log_message(
                f"Export warning: {skipped:,} molecule(s) skipped due to errors."
            )
            print(
                f"Skipped {skipped} SMILES during export: "
                f"First 20: {skipped_smiles[:20]}"
            )
        try:
            with open(os.path.join(exportPath, f"{name}_log.txt"), 'w') as ouput_log_file:
                ouput_log_file.write(self.textbox.toPlainText())
        except OSError as e:
            print(f"Could not write log file: {e}")

    def build_library_df(self, lib_name: str = "Library") -> pd.DataFrame:
        global baseSMILES

        rows = []
        batch_size = 2000
        batches = _chunk_list(baseSMILES, batch_size)

        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [executor.submit(_compute_props_batch, batch) for batch in batches]

            for future in concurrent.futures.as_completed(futures): rows.extend(future.result())

        df = pd.DataFrame(rows)
        self.library_df = df
        return df

##############################################################################
# Standalone RDKit/pandas helper functions.
##############################################################################

def _smiles_to_mol(smi: str):
        if not smi:
            return None
        try:
            mol = Chem.MolFromSmiles(smi, sanitize=True)
            if mol is None:
                return None
            Chem.SanitizeMol(mol)
            return mol
        except Exception:
            return None

def _morgan_fp(mol):
    if mol is None:
        return None
    try:
        return _MORGAN_GENERATOR.GetFingerprint(mol)
    except Exception:
        return None

def _fingerprint_batch(indexed_smiles_batch):
    results = []

    for original_index, smi in indexed_smiles_batch:
        mol = _smiles_to_mol(smi)
        if mol is None:
            continue
        fp = _morgan_fp(mol)
        if fp is None:
            continue
        arr = np.zeros((fp.GetNumBits(),),dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        results.append((original_index, arr))

    return results

def _fingerprint_matrix(smiles_list: list, batch_size: int = 2000) -> "tuple[np.ndarray, list]":
    if not smiles_list:
        return np.empty((0, 2048), dtype=np.uint8), []

    indexed_smiles = list(enumerate(smiles_list))
    batches = _chunk_list(indexed_smiles, batch_size)
    collected = []

    with concurrent.futures.ProcessPoolExecutor() as executor:

        futures = [executor.submit(_fingerprint_batch, batch) for batch in batches]

        for future in concurrent.futures.as_completed(futures):
            collected.extend(future.result())

    if not collected:
        return np.empty( (0, 2048), dtype=np.uint8), []

    collected.sort(key=lambda item: item[0])
    valid_indices = [item[0] for item in collected]
    rows = [item[1] for item in collected]

    return (np.vstack(rows), valid_indices)

def _pca_2d(X: "np.ndarray"):
    if X.shape[0] < 2:
        return (np.empty((0, 2)), np.zeros(2))

    X = X.astype(np.float32, copy=False)

    pca = PCA(n_components=2, svd_solver="randomized", random_state=42)
    coords = pca.fit_transform(X)
    return (coords, pca.explained_variance_ratio_)

def calculate_sascore(mol) -> float | None:
    if mol is None or _sa_scorer_module is None:
        return None

    try:
        mol = Chem.RemoveHs(mol)

        if hasattr(_sa_scorer_module, "calculateScore"):
            return round(float(_sa_scorer_module.calculateScore(mol)), 2)

        if hasattr(_sa_scorer_module, "scoreMol"):
            return round(float(_sa_scorer_module.scoreMol(mol)), 2)

    except Exception:
        return None

    return None

DESCRIPTOR_COLUMNS = ["MW", "logP", "logS", "TPSA", "HBA", "HBD", "RotB", "RingCount", 
                      "AromaticRings", "AtomCount", "HalogenCount", "NitrogenCount", 
                      "OxygenCount", "FormalCharge", "CanonSMILES", "SAScore"]

def _normalize_descriptor_props(props: dict) -> dict:
    normalized = dict(props)
    for col in DESCRIPTOR_COLUMNS:
        if col not in normalized:
            normalized[col] = None
    if "SMILES" not in normalized:
        normalized["SMILES"] = None
    numeric_columns = ["MW", "logP", "logS", "TPSA", "HBA", "HBD", "RotB", "RingCount", "AromaticRings", "AtomCount", "HalogenCount", "NitrogenCount", "OxygenCount", "FormalCharge", "SAScore"]
    for col in numeric_columns:
        if col in normalized and normalized[col] is not None:
            try:
                normalized[col] = float(normalized[col])
            except Exception:
                normalized[col] = None
    if normalized.get("SAScore") is not None:
        normalized["SAScore"] = round(float(normalized["SAScore"]), 2)
    return normalized


def _ensure_descriptor_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        return df
    if "SMILES" not in df.columns:
        return df  
    
    df = df.copy()
   
    for col in DESCRIPTOR_COLUMNS:
        if col not in df.columns:
            df[col] = None
    for col in ["MW", "logP", "logS", "TPSA", "HBA", "HBD", "RotB", "RingCount", "AromaticRings", "AtomCount", "HalogenCount", "NitrogenCount", "OxygenCount", "FormalCharge", "SAScore"]:
        df[col] = pd.to_numeric(df[col], errors="coerce") 
    
    df["SAScore"] = df["SAScore"].round(2)

    needs_calc = df[["MW", "logP", "logS", "TPSA", "HBA", "HBD", "RotB", "RingCount", "AromaticRings", "AtomCount", "HalogenCount", "NitrogenCount", "OxygenCount", "FormalCharge", "SAScore", "CanonSMILES"]].isna().any(axis=1)
    
    for idx, smi in df["SMILES"].items():
        if not needs_calc.loc[idx]:
            continue   
        if not isinstance(smi, str) or not smi:
            continue   
        
        mol = _smiles_to_mol(smi)
        props = _normalize_descriptor_props(_compute_props(smi) if mol is not None else {"SMILES": smi})
        
        for col in DESCRIPTOR_COLUMNS:
            if col == "CanonSMILES":
                if pd.isna(df.at[idx, col]) or df.at[idx, col] is None:
                    df.at[idx, col] = props.get(col)
            elif col == "SAScore" and pd.isna(df.at[idx, col]):
                df.at[idx, col] = props.get(col)
            elif pd.isna(df.at[idx, col]):
                df.at[idx, col] = props.get(col)
    
    return df


def _render_molecule_image(mol, width: int = 220, height: int = 140):
    if mol is None:
        return None
    try:
        from rdkit.Chem.Draw import rdMolDraw2D  
        drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
        opts = drawer.drawOptions()
        opts.addAtomIndices = False
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        png_bytes = drawer.GetDrawingText()
        image = QImage()
        image.loadFromData(png_bytes)
        if image.isNull():
            return None
        return QPixmap.fromImage(image.scaled(width, height, Qt.AspectRatioMode.KeepAspectRatio))
    except Exception:
        return None


def _format_table_value(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _set_mol_properties(mol, properties: dict):
    if mol is None:
        return
    for prop_name, prop_value in properties.items():
        if prop_name is None:
            continue
        if prop_value is None or (isinstance(prop_value, float) and math.isnan(prop_value)):
            prop_text = ""
        else:
            prop_text = str(prop_value)
        try:
            mol.SetProp(str(prop_name), prop_text)
        except Exception:
            continue

def _calculate_logs(mol) -> float:
    logp = Descriptors.MolLogP(mol)
    mw = Descriptors.MolWt(mol)
    rb = rdMolDescriptors.CalcNumRotatableBonds(mol)
    heavy = mol.GetNumHeavyAtoms()
    ap = (sum(1 for a in mol.GetAtoms() if a.GetIsAromatic()) / heavy) if heavy > 0 else 0.0
    return 0.16 - 0.63 * logp - 0.0062 * mw + 0.066 * rb - 0.74 * ap

def _compute_props(smi: str) -> dict:
    mol = _smiles_to_mol(smi)
    if mol is None:
        return {
            "SMILES": smi,
            "MW": None,
            "logP": None,
            "logS": None,
            "TPSA": None,
            "HBA": None,
            "HBD": None,
            "RotB": None,
            "RingCount": None,
            "AromaticRings": None,
            "AtomCount": None,
            "HalogenCount": None,
            "NitrogenCount": None,
            "OxygenCount": None,
            "FormalCharge": None,
            "CanonSMILES": None,
            "SAScore": None,
        }
    return {
        "SMILES": smi,
        "MW": Descriptors.MolWt(mol),
        "logP": Descriptors.MolLogP(mol),
        "logS": round(_calculate_logs(mol), 2),
        "TPSA": rdMolDescriptors.CalcTPSA(mol),
        "HBA": rdMolDescriptors.CalcNumHBA(mol),
        "HBD": rdMolDescriptors.CalcNumHBD(mol),
        "RotB": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "RingCount": rdMolDescriptors.CalcNumRings(mol),
        "AromaticRings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "AtomCount": mol.GetNumAtoms(),
        "HalogenCount": sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() in (9, 17, 35, 53, 85)),
        "NitrogenCount": sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 7),
        "OxygenCount": sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 8),
        "FormalCharge": Chem.GetFormalCharge(mol),
        "CanonSMILES": Chem.MolToSmiles(mol, canonical=True),
        "SAScore": calculate_sascore(mol),
    }

def _compute_props_batch(smiles_batch):
    return [_compute_props(smi) for smi in smiles_batch]

# --- Application entry point -------------------------------------------------
if __name__ == '__main__':
    try:
        mp.set_start_method('fork' if sys.platform == 'linux' else 'spawn')
    except RuntimeError:
        pass

    import traceback

    def _handle_uncaught_exception(exc_type, exc_value, exc_tb):
        tb_text = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        print(tb_text)
        try:
            QMessageBox.critical(None, "Unexpected error", tb_text[-2000:])
        except Exception:
            pass

    sys.excepthook = _handle_uncaught_exception

    app = QApplication(sys.argv)
    mainmenu = Main()  
    mainmenu.show()
    app.exec()
