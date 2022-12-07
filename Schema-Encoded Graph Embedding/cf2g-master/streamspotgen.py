import os
for i in range(1, 7):
    if i == 1:
        OUTDIR = "youtube"
    elif i == 2:
        OUTDIR = "gmail"
    elif i == 3:
        OUTDIR = "vgame"
    elif i == 4:
        OUTDIR = "attack"
    elif i == 5:
        OUTDIR = "download"
    elif i == 6:
        OUTDIR = "cnn"
    for j in range((i-1)*100, (i-1)*100+100):
        statement = (r"python3 cf2g.py -f streamspot -i /root/autodl-tmp/fullgraph/%s/%s.txt -g /root/autodl-tmp/PROV/data/streamspot/%s/graph%s.json"
                     % (str(i),
                       str(j),
                       OUTDIR,
                       str(j))
        )
        print(statement)
        os.system(statement)

