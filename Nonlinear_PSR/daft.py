from matplotlib import rc
import matplotlib
rc("font", family="serif", size=12)
rc("text", usetex=True)
import daft

matplotlib.rcParams['text.usetex']=False
matplotlib.rcParams['text.latex.unicode']=False

pgm = daft.PGM(shape = [7.3, 5.3], origin = [-1.5, -0.75])
pgm.add_node(daft.Node('00', ' ', 0, 3))
pgm.add_node(daft.Node('01', ' ',0, 2))
pgm.add_node(daft.Node('02', ' ',0, 1))

pgm.add_node(daft.Node('10', ' ',1, 4, observed=True))
pgm.add_node(daft.Node('11', ' ',1, 3, observed=True))
pgm.add_node(daft.Node('12', ' ',1, 2, observed=True))
pgm.add_node(daft.Node('13', ' ',1, 1, observed=True))
pgm.add_node(daft.Node('14', ' ',1, 0, observed=True))

pgm.add_node(daft.Node('20', ' ',2, 2.5))
pgm.add_node(daft.Node('21', ' ',2, 1.5))

pgm.add_node(daft.Node('30', ' ',3, 4, observed=True))
pgm.add_node(daft.Node('31', ' ',3, 3, observed=True))
pgm.add_node(daft.Node('32', ' ',3, 2, observed=True))
pgm.add_node(daft.Node('33', ' ',3, 1, observed=True))
pgm.add_node(daft.Node('34', ' ',3, 0, observed=True))

pgm.add_node(daft.Node('40', ' ',4, 3))
pgm.add_node(daft.Node('41', ' ',4, 2))
pgm.add_node(daft.Node('42', ' ',4, 1))

index = [[0, 1, 2], [0, 1,2, 3, 4], [0, 1],[0, 1,2, 3, 4], [0, 1, 2]]
for i in range(0, len(index)-1):
    for j in range(0, len(index[i])):
        for k in range(0, len(index[i+1])):
            pgm.add_edge(str(i)+str(index[i][j]), str(i+1)+str(index[i+1][k]))
pgm.add_plate(daft.Plate([-0.5, -0.4, 2.2, 4.8], label = 'P-Net'))
pgm.add_plate(daft.Plate([2.3, -0.4, 2.3, 4.8], label = 'S-Net', label_offset = [90,5]))
pgm.render()
pgm.figure.savefig("factorization.jpg", dpi = 1000)











































from matplotlib import rc
import matplotlib
rc("font", family="serif", size=12)
rc("text", usetex=True)
import daft

matplotlib.rcParams['text.usetex']=False
matplotlib.rcParams['text.latex.unicode']=False

pgm = daft.PGM(shape = [9.3, 5.3], origin = [-1.5, -0.75])
pgm.add_node(daft.Node('00', ' ',0, 3))
pgm.add_node(daft.Node('01', ' ',0, 2))
pgm.add_node(daft.Node('02', ' ',0, 1))

pgm.add_node(daft.Node('10', ' ',1, 4, observed=True))
pgm.add_node(daft.Node('11', ' ',1, 3, observed=True))
pgm.add_node(daft.Node('12', ' ',1, 2, observed=True))
pgm.add_node(daft.Node('13', ' ',1, 1, observed=True))
pgm.add_node(daft.Node('14', ' ',1, 0, observed=True))

pgm.add_node(daft.Node('20', ' ',2, 2.5))
pgm.add_node(daft.Node('21', ' ',2, 1.5))

pgm.add_node(daft.Node('30', ' ',3, 3.5, observed=True))
pgm.add_node(daft.Node('31', ' ',3, 2.5, observed=True))
pgm.add_node(daft.Node('32', ' ',3, 1.5, observed=True))
pgm.add_node(daft.Node('33', ' ',3, 0.5, observed=True))

pgm.add_node(daft.Node('40', ' ',4, 2.5))
pgm.add_node(daft.Node('41', ' ',4, 1.5))


pgm.add_node(daft.Node('50', ' ',5, 4, observed=True))
pgm.add_node(daft.Node('51', ' ',5, 3, observed=True))
pgm.add_node(daft.Node('52', ' ',5, 2, observed=True))
pgm.add_node(daft.Node('53', ' ',5, 1, observed=True))
pgm.add_node(daft.Node('54', ' ',5, 0, observed=True))

pgm.add_node(daft.Node('60', ' ',6, 3))
pgm.add_node(daft.Node('61', ' ',6, 2))
pgm.add_node(daft.Node('62', ' ',6, 1))

index = [[0, 1, 2], [0, 1,2, 3, 4], [0, 1],[0, 1, 2, 3], [0, 1], [0, 1,2, 3, 4], [0, 1, 2]]
for i in range(0, len(index)-1):
    for j in range(0, len(index[i])):
        for k in range(0, len(index[i+1])):
            pgm.add_edge(str(i)+str(index[i][j]), str(i+1)+str(index[i+1][k]))
pgm.add_plate(daft.Plate([-0.5, -0.4, 2.2, 4.8], label = 'P-Net'))
pgm.add_plate(daft.Plate([2.3, -0.4, 1.4, 4.8], label = 'Tran-Net', label_offset = [12,5]))
pgm.add_plate(daft.Plate([4.3, -0.4, 2.3, 4.8], label = 'S-Net', label_offset = [90,5]))
pgm.render()
pgm.figure.savefig("fdsa.pdf")

