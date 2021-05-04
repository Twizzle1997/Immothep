import csv
from src.classes.credentials import Credentials as cr
import os

class Splitter:

    def split_datas(self, nomfichier, columnwanted, dirname):
        
        '''
        Break raw data into many files
        '''
        


        filename = nomfichier

        os.makedirs(cr.CURATED_LOCAL_PATH + dirname + os.sep, exist_ok=True)

        csv.field_size_limit(10000000)

        with open(cr.CURATED_LOCAL_PATH + filename, encoding='utf-8') as file_stream:
            file_stream_reader = csv.DictReader(file_stream, delimiter='|')
            open_files_references = {}

            for row in file_stream_reader:
                name_of_file = row[columnwanted]

                # Open a new file and write the header
                if name_of_file not in open_files_references:
                    # print (cr.CURATED_LOCAL_PATH + dirname + os.sep() + '{}.csv'.format(name_of_file))
                    output_file = open(cr.CURATED_LOCAL_PATH + dirname + os.sep + '{}.csv'.format(name_of_file[:-5]), 'w', encoding='utf-8', newline='')
                    dictionary_writer = csv.DictWriter(output_file, fieldnames=file_stream_reader.fieldnames, delimiter='|')
                    dictionary_writer.writeheader()
                    open_files_references[name_of_file] = output_file, dictionary_writer
                # Always write the row
                open_files_references[name_of_file][1].writerow(row)
            # Close all the files
            for output_file, _ in open_files_references.values():
                output_file.close()