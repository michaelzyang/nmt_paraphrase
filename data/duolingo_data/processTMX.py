from translate.storage.tmx import tmxfile
import argparse
import sys
 
parser = argparse.ArgumentParser(add_help=False)

parser.add_argument('--filename', type=str)
parser.add_argument('--source', type=str, default=f"en")
parser.add_argument('--target', type=str, default=f"vi")
parser.add_argument('--save_dir', type=str, default=f"./")

args = parser.parse_args()

source_lang_path = args.save_dir + args.source + '.txt'
target_lang_path = args.save_dir +  args.target + '.txt'


with open(args.filename, 'rb') as inFile, open(source_lang_path, 'w+') as sourceFile, open(target_lang_path, 'w+') as targetFile:
    tmx_file = tmxfile(inFile, args.source, args.target)
    for val in tmx_file.unit_iter():
        sourceFile.write(val.getsource() + '\n')
        targetFile.write(val.gettarget() + '\n')
    