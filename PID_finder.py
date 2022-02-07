import fire
import os
import pandas as pd
from xml.dom import minidom


def finder(uniform):
    grabfood100 = pd.read_csv('pid_finder_files/grabfood100.csv')
    xmldoc = minidom.parse('pid_finder_files/all_pedsettings.xml')
    xmldoc_uniform = minidom.parse('pid_finder_files/pedsetting_uniform.xml')

    #This PID is not used in non-uniform
    UNIFORM_EXCLUDE = '96'

    uniform_hashes = {}
    non_uniform_hashes = {}
    eq_ids = {}
    PIDS = []

    uniform_pedlist = xmldoc_uniform.getElementsByTagName('PedSetting')
    for ped in uniform_pedlist:
        uniform_hashes[ped.attributes['model'].value] = ped.attributes['id'].value

    xml_pedlist = xmldoc.getElementsByTagName('PedSetting')
    for ped in xml_pedlist:
        non_uniform_hashes[ped.attributes['model'].value] = ped.attributes['id'].value
    
    for k,v in non_uniform_hashes.items():
        if k in uniform_hashes.keys():
            eq_ids[uniform_hashes[k]] = v
    
    del(eq_ids[UNIFORM_EXCLUDE])
    if uniform == True:
        PIDS = eq_ids.keys()
    else:
        PIDS = eq_ids.values()

    # for ped in xml_pedlist:
    #     if ped.attributes['model'].value in hashes:
    #         PIDS.append(ped.attributes['id'].value)
    # print(PIDS)
    # print(f'{len(PIDS)=}')
    return PIDS

if __name__=='__main__':
    fire.Fire(finder)