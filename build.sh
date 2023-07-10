#python3 -m pip install --upgrade build
#python3 -m pip install --upgrade twine

#rm dist -rf
#rm *.egg-info -rf

python3 -m build
python3 -m twine upload dist/*