import taichi as ti
ti.init(arch=ti.cpu)

@ti.kernel
def hello():
    print("Hello,2026Taichi!")
hello()
