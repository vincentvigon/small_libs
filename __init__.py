import sys
"ajouter un chemin au début de la liste"

sys.path.insert(0, "small_libs")

from consecutive_windows import make_consecutive_windows

print("importation de small_libs, ajout de small_libs dans votre PYTHONPATH")
