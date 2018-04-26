import re
contents_dw = [ ]
contents_dw_dict = { }
with open("ILMDOS.A1.DW.dat","r") as f:
    for lines in f.readlines():
        line = lines.strip().split()
        contents_dw.append(line)
#print(contents_dw)
contents_dw[0].pop(0)
#print(contents_dw[0])
for i in range(len(contents_dw[0])):
    contents_dw[0][i] =contents_dw[0][i]+"_dw"
#print(contents_dw[0])
con_dw = zip(contents_dw[0],contents_dw[-1])
#print(con_dw)
for i in con_dw:
    contents_dw_dict[i[0]]=i[-1]
#print(contents_dw_dict)

contents_up = [ ]
contents_up_dict = { }
with open("ILMDOS.A1.UP.dat","r") as f:
    for lines in f.readlines():
        line = lines.strip().split()
        contents_up.append(line)
#print(contents_up)
contents_up[0].pop(0)
#print(contents_up[0])
for i in range(len(contents_up[0])):
    contents_up[0][i] =contents_up[0][i]+"_up"
#print(contents_up[0])
con_up = zip(contents_up[0],contents_up[-1])
#print(con_up)
for i in con_up:
    contents_up_dict[i[0]]=i[-1]
#print(contents_up_dict)
total = { }
for i in range(1,len(con_up)):
    st = contents_up[0][i][0:-3]
    total[st]=(float(con_up[i][-1])-float(con_dw[i][-1]))
print(total)
total_elec=0
d_elec=0
pattern = re.compile(r'd*')
for key,value in total.items():
    if pattern.findall(key)[0]=='d':
        d_elec+=value 
print("d  electrons:",d_elec)
for key,value in total.items():
    total_elec+=value
print("total electrons:",total_elec)
