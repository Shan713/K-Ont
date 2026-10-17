"""Load a downloaded pickle that should hold only plain data (lists, tuples, str, int, float...).
pickle.load would run whatever the file asks for; this Unpickler refuses every global (class or function
reference), so a file that is not plain data raises instead of executing anything."""
import gzip
import pickle


class _PlainOnly(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"refused global {module}.{name}: not plain data")


def load(path):
    with (gzip.open(path, "rb") if path.endswith(".gz") else open(path, "rb")) as f:
        return _PlainOnly(f).load()
