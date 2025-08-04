

# Usage: This program aims to transfer your markdown file into a way zhihu.com can recognize correctly.
#        It will mainly deal with your local images and the formulas inside.

import os, re
import argparse
import subprocess
import chardet
import functools
import os.path as op

from PIL import Image
from pathlib import Path

###############################################################################################################
## Please change the GITHUB_REPO_PREFIX value according to your own GitHub user name and repo name. ##
###############################################################################################################
# GITHUB_REPO_PREFIX = "https://raw.githubusercontent.com/`YourUserName`/`YourRepoName`/main/"
# Your repo remote link (without Data folder)
GITHUB_REPO_PREFIX = "http://113.44.140.251:9000/junk/Discrete-Distribution-Networks.github.io/"
COMPRESS_THRESHOLD = 5e5 # The threshold of compression

# The main function for this program
def process_for_zhihu():

    if args.encoding is None:
        with open(str(args.input), 'rb') as f:
            s = f.read()
            chatest = chardet.detect(s)
            args.encoding = chatest['encoding']
        print(chatest)
    with open(str(args.input),"r",encoding=args.encoding) as f:
        lines = f.read()
        lines = image_ops(lines)
        lines = formula_ops(lines)
        lines = table_ops(lines)
        output_file = op.join(args.file_parent, args.input.stem+"_for_zhihu.md")
        with open(output_file, "w+", encoding=args.encoding) as fw:
            fw.write(lines)
        print(f"Output file created: {output_file}")

# Deal with the formula and change them into Zhihu original format
def formula_ops_old(_lines):
    _lines = re.sub('((.*?)\$\$)(\s*)?([\s\S]*?)(\$\$)\n', '\n<img src="https://www.zhihu.com/equation?tex=\\4" alt="\\4" class="ee_img tr_noresize" eeimg="1">\n', _lines)
    _lines = re.sub('(\$)(?!\$)(.*?)(\$)', ' <img src="https://www.zhihu.com/equation?tex=\\2" alt="\\2" class="ee_img tr_noresize" eeimg="1"> ', _lines)
    return _lines

def formula_ops(_lines):
    # _lines = re.sub('((.*?)\$\$)(\s*)?([\s\S]*?)(\$\$)\n', '\n<img src="https://www.zhihu.com/equation?tex=\\4" alt="\\4" class="ee_img tr_noresize" eeimg="1">\n', _lines)
    _lines = re.sub('(\$)(?!\$)(.*?)(\$)', ' $$\\2$$ ', _lines)
    return _lines

# The support function for image_ops. It will convert relative paths to GitHub absolute URLs
def rename_image_ref(m, original=True):
    ori_path = m.group(2) if original else m.group(1)
    
    # Remove any leading ./ from the path
    if ori_path.startswith('./'):
        ori_path = ori_path[2:]
    
    # Check if the image file exists relative to the markdown file
    full_local_path = op.join(args.file_parent, ori_path)
    if not op.exists(full_local_path):
        print(f"Warning: Image file not found: {full_local_path}")
        return m.group(0)  # Return original if file doesn't exist
    
    # Convert relative path to GitHub absolute URL
    github_url = GITHUB_REPO_PREFIX + ori_path
    
    print(f'Local path: {full_local_path}')
    print(f'GitHub URL: {github_url}')
    
    if original:
        return "!["+m.group(1)+"]("+github_url+")"
    else:
        return '<img src="'+github_url+'"'

def cleanup_image_folder():
    # This function is no longer needed since we don't copy files
    pass

# Search for the image links which appear in the markdown file. It can handle two types: ![]() and <img src="LINK" alt="CAPTION" style="zoom:40%;" />.
# The second type is mainly for those images which have been zoomed.
def image_ops(_lines):
    _lines = re.sub(r"\!\[(.*?)\]\((.*?)\)",functools.partial(rename_image_ref, original=True), _lines)
    _lines = re.sub(r'<img src="(.*?)"',functools.partial(rename_image_ref, original=False), _lines)
    return _lines

# Deal with table. Just add a extra \n to each original table line
def table_ops(_lines):
    return re.sub("\|\n",r"|\n\n", _lines)


def reduce_single_image_size(image_path):
    # This function is kept for potential future use but not currently used
    # since we're not copying/modifying images
    output_path = Path(image_path).parent/(Path(image_path).stem+".jpg")
    if op.exists(image_path):
        img = Image.open(image_path)
        if(img.size[0]>img.size[1] and img.size[0]>1920):
            img=img.resize((1920,int(1920*img.size[1]/img.size[0])),Image.ANTIALIAS)
        elif(img.size[1]>img.size[0] and img.size[1]>1080):
            img=img.resize((int(1080*img.size[0]/img.size[1]),1080),Image.ANTIALIAS)
        img.convert('RGB').save(output_path, optimize=True,quality=85)
    return output_path

# Push your new change to github remote end
def git_ops():
    subprocess.run(["git","add","-A"])
    subprocess.run(["git","commit","-m", "update file "+args.input.stem])
    subprocess.run(["git","push", "-u", "origin", "master"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Please input the file path you want to transfer using --input=""')
    parser.add_argument('--compress', action='store_true', help='Compress the image which is too large (currently not used)')
    parser.add_argument('-i', '--input', type=str, help='Path to the file you want to transfer.')
    parser.add_argument('-e', '--encoding', type=str, help='Encoding of the input file')

    args = parser.parse_args()
    if args.input is None:
        raise FileNotFoundError("Please input the file's path to start!")
    else:
        args.input = Path(args.input)
        args.file_parent = str(args.input.parent)
        
        print(f"Processing file: {args.input}")
        print(f"Output will be saved to: {args.file_parent}")
        process_for_zhihu()