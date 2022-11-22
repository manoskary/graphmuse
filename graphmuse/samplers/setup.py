from distutils.core import setup, Extension

def main():
    setup(name="graphmuse",
          version="1.0.0",
          ext_modules=[Extension("graphmuse", ["graphmusemodule.c"])])

if __name__ == "__main__":
    main()

