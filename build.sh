#!/bin/sh
flatpak-builder --install --user build-dir org.moschittolab.MOLDE.json --force-clean
flatpak run org.moschittolab.MOLDE
