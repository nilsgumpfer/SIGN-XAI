#python3 -m pip install --upgrade build
#python3 -m pip install --upgrade twine

rm dist -rf
rm *.egg-info -rf

python3 -m build
python3 -m twine upload dist/* -u __token__ -p pypi-AgEIcHlwaS5vcmcCJDNhMjJmZDhhLTk3ZTItNDY3ZC1iZDViLTE2ZDIxN2YyOTk1MQACD1sxLFsic2lnbnhhaSJdXQACLFsyLFsiZDYyM2NiZWMtOTliZS00ODY4LWIyZTctNDJkMDZiMjgzZjNlIl1dAAAGIFXiHjI0qhTO1TfrbkFK3YPKibhk9XviElpzkS2yJJ15