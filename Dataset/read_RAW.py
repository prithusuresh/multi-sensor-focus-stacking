import numpy as np
import matplotlib.pyplot as plt
import cv2
plt.rcParams["figure.figsize"] = [10,10]

def bytes_from_file(filename, chunksize=8192):
    with open(filename, "rb") as f:
        while True:
            chunk = f.read(chunksize)
            if chunk:
                for b in chunk:
                    yield b
            else:
                break
def read_raw(path):
    inp = []
    for b in bytes_from_file(path):
        inp.append(b)

    from copy import copy
    mipi_packed = copy(inp)

    R  = []
    Gr = []
    Gb = []
    B  = []



    mipi_dummy_removed = []
    WIDTH = 4000
    HEIGHT = 3000
    ix = 0
    for i in range(HEIGHT):
        total = []
        if i%2:
            # Gb and B
            gb = []
            b = []
            for j in range(WIDTH//4):
                residue = mipi_packed[ix+4]

                total.append(mipi_packed[ix]*4 + residue%4)
                gb.append(mipi_packed[ix]*4 + residue%4)

                residue//=4

                total.append(mipi_packed[ix+1]*4 + residue%4)
                b.append(mipi_packed[ix+1]*4 + residue%4)

                residue//=4

                total.append(mipi_packed[ix+2]*4 + residue%4)
                gb.append(mipi_packed[ix+2]*4 + residue%4)

                residue//=4

                total.append(mipi_packed[ix+3]*4 + residue%4)
                b.append(mipi_packed[ix+3]*4 + residue%4)

                residue//=4
                ix += 5
            Gb.append(gb)
            B.append(b)
        else:
            ## R and Gr
            r = []
            gr = []
            for j in range(WIDTH//4):
                residue = mipi_packed[ix+4]

                total.append(mipi_packed[ix]*4 + residue%4)
                r.append(mipi_packed[ix]*4 + residue%4)

                residue//=4

                total.append(mipi_packed[ix+1]*4 + residue%4)
                gr.append(mipi_packed[ix+1]*4 + residue%4)

                residue//=4

                total.append(mipi_packed[ix+2]*4 + residue%4)
                r.append(mipi_packed[ix+2]*4 + residue%4)

                residue//=4

                total.append(mipi_packed[ix+3]*4 + residue%4)
                gr.append(mipi_packed[ix+3]*4 + residue%4)

                residue//=4
                ix += 5
            R.append(r)
            Gr.append(gr)
        ix += 8 ## dummy bytes
        mipi_dummy_removed.append(total)
    bin_image = np.array(mipi_dummy_removed, dtype = np.uint16)
    return bin_image
def demosaic(image):
    bin_image = cv2.cvtColor(image, cv2.COLOR_BayerGR2RGB)
    return bin_image

def main():
    import os
    r = 0
    for i in os.listdir(os.getcwd()):
        if "RAW" not in i:
            continue
        RAW = read_raw(i)
        to_8bit = lambda t: int(t*(255/1023))
        vfunc = np.vectorize(to_8bit, otypes=[np.uint8])
        int8 = vfunc(RAW)
        int8 = cv2.cvtColor(demosaic(int8), cv2.COLOR_RGB2BGR)
        name = i.split(".")[0]
        plt.imsave(str(name)+".png",int8)
        plt.show()
        r+=1
        break

if __name__=="__main__":
    main()