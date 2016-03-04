import os
import csv
import time

""" Process original MIMIC CSV Files to reorganize the data. """


def csv_length(src, hdr=False):
    """ Return number of data rows in a csv, optionally including header """
    with open(src) as f:
        for i, l in enumerate(f):
            pass
    return i + hdr


def csv_hist(src, dst, col_target):
    """ Output a file with count occurrences (eg histogram)
    of unique elements in specified column.

    Args:
        col_target (string): Header of column to count.
        dst (string) : Name of output file. """
    dict_counts = {}
    with open(src) as f:
        i = 0
        for line in f:
            # Convert line (string) to list of strings
            line = line.strip('"').split(',')

            # Find column index
            if i == 0:
                for j, col_name in enumerate(line):
                    if col_target in col_name:
                        col_idx = j
                        break

            # Continue with dictionary building / histogram
            else:
                key = int(line[col_idx])
                dict_counts[key] = dict_counts.get(key, 0) + 1

            # Show progress
            i += 1
            if i % 5000 == 0:
                print(i)

    # Save dictionary to CSV
    writer = csv.writer(open(dst, 'w', newline=''))
    for key, value in dict_counts.items():
        writer.writerow([key, value])


def csv_2_minicsv(src, num_lines):
    """ Create a CSV with the first n data rows of source csv.
    The header row is included and doesn't count towards row count.
    This function is primarily for testing. """
    dst = os.path.splitext(src)[0] + '_mini.csv'
    with open(src) as f_src:
        i = 0
        with open(dst, 'w') as f_dst:
            for line in f_src:
                f_dst.write(line)
                i += 1
                if i == num_lines + 1:
                    break


def chart_2_patients(src, dpath_dst, start=0, num_lines=263201375):
    """ Create individual files per patient per admission from CHARTEVENTS.
        Filename format: [p_id]_[hadm_id].csv """
    with open(src, 'r') as src:
        i, skip_count = 0, 0
        next(src)
        h_id_prev, fout = None, None
        step = 2000
        prog_pts = [x * step for x in range(num_lines // step)]

        for line in src:
            if skip_count < start:
                skip_count += 1
                pass
            else:
                line_list = line.split(',')
                p_id = line_list[1]    # string of patient id
                h_id = line_list[2]    # string of hadm_id

                if h_id != h_id_prev:  # Close file only if different hadm_id
                    if i > 0:
                        fout.close()
                    p_fname = p_id + '_' + h_id + '.csv'
                    fout = open(os.path.join(dpath_dst, p_fname), 'a')
                    h_id_prev = h_id

                fout.write(line)

                if i in prog_pts:      # Show progress in command prompt
                    print("%.4f %%" % (prog_pts.index(i) / len(prog_pts)))
                i += 1
                if i == num_lines & num_lines > 0:
                    break
    print('Total rows processed: %s' % (start + i))


def csv_2_dict(src, key_idx, val_idx):
    """ Return a dictionary for lookups using a csv.
    Key and values columns in file are specifed by two indexes. """
    dict_lu = {}
    with open(src, 'r') as f_src:
        next(f_src)
        for line in f_src:
            line = line.split(',')
            key = line[key_idx]
            val = line[val_idx]

            if val:
                dict_lu[key] = val
    return dict_lu


def unify_csv(src_csv, src_lu, unify_idx, lu_key_idx, lu_val_idx):
    """ Create a new version of a csv where elements of a column have been unified
    based on a lookup table """
    lu = csv_2_dict(src_lu, lu_key_idx, lu_val_idx)

    with open(src, 'r') as f_src:
        next(f_src)

        for line in f_src:
            line = line.split(',')
            val_ori = line[unify_idx]
            val_new = lu.get(val_ori, val_ori)
            line[unify_idx] = val_new




t0 = time.clock()

fname_chart = 'C:/Users/Mark/Downloads/MIMIC Data/Raw Data CSVs/CHARTEVENTS_DATA_TABLE.csv'
fname_chart_mini = 'C:/Users/Mark/Downloads/MIMIC Data/Raw Data CSVs/CHARTEVENTS_DATA_TABLE_mini.csv'
fname_chart_hist = 'C:/Users/Mark/Downloads/MIMIC Data/Data Summary/ChartEvents Hist - ItemID.csv'
fname_lu_items = 'C:/Users/Mark/Downloads/MIMIC Data/Lookups/Lookup_D_ITEMS.csv'
dpath_patientfiles = 'C:/Users/Mark/Downloads/MIMIC Data/Patient Data'

# num_lines = csv_length(fname_chart)  # 263201375
# csv_2_minicsv(fname_chart, 30)
# csv_hist(fname_chart, fname_chart_hist, 'ITEMID')
# chart_2_patients(
#     fname_chart, dpath_patientfiles, 1000000, 1000000)
unify_csv(fname_chart_mini, fname_lu_item, 1, 2)
t1 = time.clock()
print("Run time: %.4f" % (t1 - t0))
