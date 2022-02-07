import os

path = '/home/GTA/GTA_Long'
file = 'query_1.txt'
with open(os.path.join(path,file),'r') as f:
    lines = f.readlines()
    tots = {}
    for line in lines:
        lines = line.strip('\n') 
        slash_split = line.split('/')
        if slash_split[0] not in tots:
            tots[slash_split[0]] = 1
        else:
            tots[slash_split[0]]+=1
    for k,v in tots.items():
        if v <10:
            print(k,v)