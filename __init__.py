import sys
"ajouter un chemin au début de la liste"

sys.path.insert(0, "small_libs")

print("importation de small_libs, ajout de small_libs dans votre PYTHONPATH")
print("small_libs importation")

from consecutive_windows.make_consecutive_windows import  make_consecutive_windows
from derivator_fft.derivator import Derivator_fft
from derivator_fft.fft_nd import Fft_nd, fft_nd, ifft_nd






