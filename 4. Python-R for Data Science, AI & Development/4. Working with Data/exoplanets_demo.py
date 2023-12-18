import requests
import pandas as pd
payload: dict = {
    "workspace":
        "2023.12.01_05.17.37_028941/TblView/2023.12.01_05.17.37_028941",
    "useTimestamp": 1,
    "table": "/exodata/kvmexoweb/ExoTables/PS.tbl",
    "format": "CSV",
    "user": "",
    "label": "",
    "columns": "pl_name_display,hostname,default_flag,sy_snum,sy_pnum,"
               "discoverymethod,disc_year,disc_facility,soltype,"
               "pl_controv_flag,pl_refname,pl_orbperstr,pl_orbsmaxstr,"
               "pl_radestr,pl_radjstr,pl_bmassestr,pl_bmassjstr,"
               "pl_bmassprov,pl_orbeccenstr,pl_insolstr,pl_eqtstr,"
               "ttv_flag,st_refname,st_spectype,st_teffstr,st_radstr,"
               "st_massstr,st_metstr,st_metratio,st_loggstr,sy_refname,rastr,"
               "decstr,sy_diststr,sy_vmagstr,sy_kmagstr,sy_gaiamagstr,"
               "rowupdate,"
               "pl_pubdate,releasedate",
    "rows": "both",
    "mission": "ExoplanetArchive"}

url: str = \
    "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/IceTable/" \
    "nph-iceTblDownload"


request: requests.Response = \
    requests.post(url, data=payload,
                  headers={"content-type": "application/json"})
print(request.status_code)
csv_string = request.text
csv_lines: [str] = []
with open("./exoplanets_NASA.csv", "w") as exoplanets_file:
    for line in csv_string.split("\n"):
        if line.startswith("#"):
            pass
        else:
            exoplanets_file.write(line)
            print(line)
    exoplanets_file.close()

print("FGFGFFGFGF")
