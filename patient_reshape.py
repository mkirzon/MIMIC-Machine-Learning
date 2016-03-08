import csv
import numpy as np
import pdb
import pandas



def Read_unique_item(unique_itemid_csv, hdr=False):
    """Input a csvfile with only a single column containing all the possible unique itemids.
    Output a python list that contains all the unique itemids"""
    
    with open(unique_itemid_csv) as f:
        temp = f.read().splitlines()
    
    unique_itemids = [int(x) for x in temp[1:]] if hdr==True else [int(x) for x in temp] 
                
    return unique_itemids
    
    
def Dict_unique_item(unique_itemids):
    """Create a python dictionary that maps itemids to their row index. 
    Input unique itemids list, output dctionary.
    The key, itemid value, will be arranged in ascending order,
    but the value, the row index, will not"""
    
    item_row_dict = {}
    
    for ele_count, ele in enumerate(unique_itemids):
        item_row_dict[ele] = ele_count
        
    return item_row_dict


def Reshape_patient(itemid2Index_dict, uni_itemid_lu, patient_original_data, hdr = False):
    
    
    itemidSize = len(itemid2Index_dict)
    temp = patient_original_data
    timeStampsList = []
    
    #Create a 2d matrix that now has a width of itemid size but 0 length.
    reshaped_patient_mat = np.empty(shape = [0,itemidSize])
     
    # tuple = (itemID, chartTime, valueNUM, valueUOM)
    temp_tuple = [(item[4], item[5], item[9], item[10]) for item in temp[1:]]\
    if hdr == True else [(item[4], item[5], item[9], item[10]) for item in temp]
    
    
    for x in temp_tuple:
        if x[1] == timeStampsList[-1]:
            
            uni_itemid = uni_itemid_lu[ x[0] ]
            uni_itemid_index = itemid2Index_dict[ uni_itemid ]
            
            reshaped_patient_mat[-1][ uni_itemid_index ] = x[2]
            
        else:
            
            uni_itemid = uni_itemid_lu[ x[0] ]
            uni_itemid_index = itemid2Index_dict[ uni_itemid ]
            
            newRow = np.zeros(itemidSize)
            newRow[ uni_itemid_index ] = x[2]
            
            
            np.append(reshaped_patient_mat, newRow, axis=0)
            timeStampsList.append(x[1])
            
    
    return reshaped_patient_mat
            
            
                 
    

unique_itemid_csv = 'unique_itemid.csv'
patient_original_csv = 'CHARTEVENTS_DATA_TABLE_mini.csv'

unique_itemids = Read_unique_item(unique_itemid_csv,hdr=True)
unique_itemids_dict = Dict_unique_item(unique_itemids)
#tu = Reshape_patient(unique_itemids_dict,patient_original_csv, hdr = True)
