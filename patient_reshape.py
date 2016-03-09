import csv
import numpy as np
import pdb
import pandas
import os
from datetime import datetime



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


def Reshape_patient( uni_itemid_lu, itemid2Index_dict, patient_original_data):
    """
    Input: 
        1. Unique item id look up dictionay. Key: original item ID's. Value: unique item ID's
        2. Index look up dictionary, Key: unique item ID's. Value: Index number, which is later used as row index
        3. Original patient data, in form of list of lists. Rows are differnt items sorted by time stamps already.
        Every elements are values.
        4. Boolen, if there are header in the file. Default: False
        
        
        """
    
    itemidSize = len(itemid2Index_dict)
    temp = patient_original_data
    timeStampsList = []
    
    #Create a 2d matrix that now has a width of itemid size but 0 length.
    reshaped_patient_mat = np.empty(shape = [1,itemidSize])
     
    # tuple = (itemID, chartTime, valueNUM, valueUOM)
    temp_tuple = [(int(item[4]), item[5], float(item[9]), item[10]) for item in temp]
    
    
    for x in temp_tuple:
        #print(x)
        if len(timeStampsList) == 0:
            
            uni_itemid = uni_itemid_lu[ x[0] ]
            uni_itemid_index = itemid2Index_dict[ uni_itemid ]
            
            newRow = np.full(itemidSize,None)
            newRow[ uni_itemid_index ] = x[2]
            print(len(newRow))
            print(itemidSize)
            
            reshaped_patient_mat[0] = newRow
            timeStampsList.append(x[1])
            print(timeStampsList)
        
        elif x[1] == timeStampsList[-1]:
            
            uni_itemid = uni_itemid_lu[ x[0] ]
            uni_itemid_index = itemid2Index_dict[ uni_itemid ]
            
            reshaped_patient_mat[-1][ uni_itemid_index ] = x[2]

            
        else:
            
            uni_itemid = uni_itemid_lu[ x[0] ]
            uni_itemid_index = itemid2Index_dict[ uni_itemid ]
            
            newRow = np.full(itemidSize,None)
            newRow[ uni_itemid_index ] = x[2]
            
            
            reshaped_patient_mat = np.append(reshaped_patient_mat, [newRow], axis=0)
            timeStampsList.append(x[1])
            
    
    reshaped_patient_mat = np.transpose(reshaped_patient_mat)
    return reshaped_patient_mat
    
def make_lu_csv(src_csv, id_idx, name_idx, has_header):
    """ 
    Input: Item ID's table with item name
    
    Create 2 files:
        1) CSV that has unified ITEMID's based on the unified name column. 
        The id corresponding to first unique name encountered is used as the unifying id. 
        2) CSV with unique ID file. """

    #dst = os.path.splitext(src_csv)[0] + '_lu.csv'
    #f_dst = open(dst, 'w', newline='')
    dst1 = 'ExtraColWithUniqueID.csv'
    f_dst = open(dst1, 'w')
    
    
    #f_unique = open(os.path.dirname(src_csv) + '/ITEMID_Unique.csv', 'w')
    
    dst2 = 'UniqueIDLookUp.csv'
    f_unique = open(dst2, 'w')
    writer = csv.writer(f_dst,lineterminator='\n')


    uni_itemid_list =[]
    
    with open(src_csv, 'r') as f_src:
        name_pr = ''
        if has_header:
            f_dst.write(f_src.readline())
        for line in f_src:
            line = line.rstrip().split(',')
            name = line[name_idx]
            id_ori = line[id_idx]
            if name_pr == '' or name != name_pr:
                id_univ = id_ori
                f_unique.write(id_univ + '\n')
                uni_itemid_list.append(int(id_univ))
            line.insert(name_idx, id_univ)
            writer.writerow(line)
            name_pr = name
            
    return uni_itemid_list
          
                        
def csv_2_dict(src_csv, key_idx, val_idx, header):
    """ Return a dictionary for lookups using a csv.
    Key and values columns in file are specifed by two indexes. """
    dict_lu = {}
    with open(src_csv, 'r') as f_src:
        if header:
            next(f_src)
        for line in f_src:
            line = line.split(',')
            key = line[key_idx]
            val = line[val_idx]

            if val:
                dict_lu[int(key)] =int(val)
    return dict_lu     
            
def patient_file_process(src_csv):
    f_in = csv.reader(open(src_csv, 'r'))
    
    # Sort data by date
    data = sorted(f_in, key=lambda row: datetime.strptime(row[5], "%Y-%m-%d %H:%M:%S"))

    # Convert dates to relative times
    #t0 = datetime.strptime(data[0][5], "%Y-%m-%d %H:%M:%S")
    #for row in data:
        #t = datetime.strptime(row[5], "%Y-%m-%d %H:%M:%S")
        #tf = t - t0
        #row[5] = tf.total_seconds() / 60
          
    return data       
    
itemid_name_csv = 'MIMIC Variables Unification Table.csv'
unique_itemid_csv = 'sample_files/unique_itemid_test.csv'
patient_original_csv = 'sample_files/sample_patient_file.csv'
uni_itemids_lu = 'ExtraColWithUniqueID.csv'

#unique_itemids = Read_unique_item(unique_itemid_csv,hdr=True)
#itemids2Index_dict = Dict_unique_item(unique_itemids)
unique_itemids = make_lu_csv(itemid_name_csv, 1, 2,  True)
uni_itemids_dict = csv_2_dict(uni_itemids_lu, 1, 2, True)
itemids2Index_dict = Dict_unique_item(unique_itemids)
original_patient_data = patient_file_process(patient_original_csv)

reshaped_patient_mat = Reshape_patient(uni_itemids_dict, itemids2Index_dict, original_patient_data)


#tu = Reshape_patient(unique_itemids_dict,patient_original_csv, hdr = True)
