import csv
from classes.credentials import Credentials as cr
import os

class Splitter:

    def split_datas(self, filename, columnwanted, dirname, sep):
        
        '''
        Break raw data into many files
        '''

        os.makedirs(cr.CURATED_LOCAL_PATH + dirname + os.sep, exist_ok=True)

        csv.field_size_limit(10000000)

        with open(cr.CURATED_LOCAL_PATH + filename, encoding='utf-8') as file_stream:
            file_stream_reader = csv.DictReader(file_stream, delimiter=',')
            open_files_references = {}

            for row in file_stream_reader:
                name_of_file = row[columnwanted]

                # Open a new file and write the header
                if name_of_file not in open_files_references:
                    output_file = open(cr.CURATED_LOCAL_PATH + dirname + os.sep + '{}.csv'.format(name_of_file), 'w', newline="", encoding='utf-8')
                    dictionary_writer = csv.DictWriter(output_file, fieldnames=file_stream_reader.fieldnames, delimiter=sep)
                    dictionary_writer.writeheader()
                    open_files_references[name_of_file] = output_file, dictionary_writer
                # Always write the row
                open_files_references[name_of_file][1].writerow(row)
            # Close all the files
            for output_file, _ in open_files_references.values():
                output_file.close()