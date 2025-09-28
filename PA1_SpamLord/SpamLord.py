"""
Created on Wed Sep 14

@author: vamsi
"""

import sys
import os
import re
import pprint

# We are splitting different types of eMails patterns into 4 types in which two are corner cases.
email_regex = []
email_regex.append('([\w]+[\w._+-]+[\w]+)\s*@\s*([\w]+[\w._+-]+[\w]+).(edu|com|in|net|org)')
email_regex.append('([\w]+[\w._+-]+[\w]+)\s*at\s*([\w]+[\w._+-]+[\w]+)\s*dot\s*(edu|com|in|net|org)')
email_regex.append('([\w.]+)\s*at\s*([\w.]+)\s*dot\s*com')
email_regex.append('([\w.]+)\s*[<\[\(]?at[>\]\)]?\s*([\w.]+)\s*[<\[\(]?dot[>\]\)]?\s*(edu|com)')
email_regex.append('([\w.]+)\s*[<\[]?@[>\]]?\s*([\w.]+)\s*[<\[]?dot[>\]]?\s*(edu|com)')
email_regex.append('([\w.]+)\s<[\w.]+>\s([\w.]+)\.(com|edu)')
email_regex.append('([\w.]+)&#x40;([\w.]+)\.(com|edu)')
email_regex.append('([\w.]+)\s*<[\w.]+>\s*@([\w.]+)\@(com|edu)')
email_regex.append('([\w.]+) \(at\) ([\w.]+) \(dot\) (com|edu)')
email_regex.append('([\w.]+) \[at] ([\w.]+)\.(edu|com)')
email_regex.append('([\w.]+)%40([\w.]+)\.(com|edu)')
email_regex.append('([\w.]+) at ([\w.]+)\.(edu|com)')
email_regex.append('([\w.]+)<[\w.]+>@([\w.]+)\.(edu|com)') 
email_regex.append('([\w.]+)\s<[\w.]+>\s([\w.]+)\.(edu|com)')


email_regex_cornercase1 = []
email_regex_cornercase1.append('write_mail\("([\w.]+)","([\w.]+)"\)')     # write_mail("tamu.edu","amato")

email_regex_cornercase2 = []
email_regex_cornercase2.append('([\w.]+)\s*at\s*([\w.]+)\s*dot\s*([\w.]+)\s*dot\s*edu')  # Captures the case with 2 dots
email_regex_cornercase2.append('([\w.]+) \[please add at sign here] ([\w.]+) \[dot] ([\w.]+) \[dot] edu')  # Captures the case with please add at sign here string


phone_regex = []
phone_regex.append("^((?!fax).)*\D(\d{3})[-\.\s\/\r\n]+(\d{3})[-\.\s\/\r\n]+(\d{4})\D.*$")
phone_regex.append("^((?!fax).)*\D\((\d{3})\)\s*(\d{3})[-\.\s](\d{4})\D.*$")
phone_regex.append(("^()\((\d{3})\)\s*(\d{3})[-\.\s](\d{4}).*$"))



""" 
TODO
This function takes in a filename along with the file object (actually
a StringIO object) and
scans its contents against regex patterns. It returns a list of
(filename, type, value) tuples where type is either an 'e' or a 'p'
for e-mail or phone, and value is the formatted phone number or e-mail.
The canonical formats are:
     (name, 'p', '###-###-#####')
     (name, 'e', 'someone@something')
If the numbers you submit are formatted differently they will not
match the gold answers

NOTE: *don't change this interface*

NOTE: You shouldn't need to worry about this, but just so you know, the
'f' parameter below will be of type StringIO. So, make
sure you check the StringIO interface if you do anything really tricky,
though StringIO should support most everything.
"""

def process_file(name, f):
    # note that debug info should be printed to stderr
    # sys.stderr.write('[process_file]\tprocessing file: %s\n' % (path))
    res = []
    pre=""
    for line in f:
        temp = line
        line = pre + " " + line
        pre = temp.replace("\n", "")
        line = line.lower()

        for pattern in email_regex:
            matches = re.findall(pattern,line)
            for m in matches:
                email = '%s@%s.%s' % m
                res.append((name,'e',email))
          
        for cornercase in email_regex_cornercase1:
            matches = re.findall(cornercase,line)
            for m in matches:
                email = '%s@%s' % (matches[0][1],matches[0][0])
                res.append((name,'e',email))
    
        for cornercase in email_regex_cornercase2:
            matches = re.findall(cornercase,line)
            for m in matches:
                email = '%s@%s.%s.edu' % m
                res.append((name,'e',email))
        
        for pattern in phone_regex:
            matches = re.findall(pattern, line)
            for m in matches:
                m = m[1:]
                result = ''.join([a + '-' for a in m])
                phone = result[:-1]
                res.append((name,'p',phone))
            
    return res

"""
You should not need to edit this function, nor should you alter
its interface
"""
def process_dir(data_path):
    # get candidates
    guess_list = []
    for fname in os.listdir(data_path):
        if fname[0] == '.':
            continue
        path = os.path.join(data_path,fname)
        f = open(path,'r', encoding = "ISO-8859-1")
        f_guesses = process_file(fname, f)
        guess_list.extend(f_guesses)
    return guess_list

"""
You should not need to edit this function.
Given a path to a tsv file of gold e-mails and phone numbers
this function returns a list of tuples of the canonical form:
(filename, type, value)
"""
def get_gold(gold_path):
    # get gold answers
    gold_list = []
    f_gold = open(gold_path,'r')
    for line in f_gold:
        gold_list.append(tuple(line.strip().split('\t')))
    return gold_list

"""
You should not need to edit this function.
Given a list of guessed contacts and gold contacts, this function
computes the intersection and set differences, to compute the true
positives, false positives and false negatives.  Importantly, it
converts all of the values to lower case before comparing
"""
def score(guess_list, gold_list):
    guess_list = [(fname, _type, value.lower()) for (fname, _type, value) in guess_list]
    gold_list = [(fname, _type, value.lower()) for (fname, _type, value) in gold_list]
    guess_set = set(guess_list)
    gold_set = set(gold_list)

    tp = guess_set.intersection(gold_set)
    fp = guess_set - gold_set
    fn = gold_set - guess_set

    pp = pprint.PrettyPrinter()
    #print('Guesses (%d): ' % len(guess_set))
    #pp.pprint(guess_set)
    #print('Gold (%d): ' % len(gold_set))
    #pp.pprint(gold_set)
    print('True Positives (%d): ' % len(tp))
    pp.pprint(tp)
    print('False Positives (%d): ' % len(fp))
    pp.pprint(fp)
    print('False Negatives (%d): ' % len(fn))
    pp.pprint(fn)
    print('Summary: tp=%d, fp=%d, fn=%d' % (len(tp),len(fp),len(fn)))

"""
You should not need to edit this function.
It takes in the string path to the data directory and the
gold file
"""
def main(data_path, gold_path):
    guess_list = process_dir(data_path)
    gold_list =  get_gold(gold_path)
    score(guess_list, gold_list)

"""
commandline interface takes a directory name and gold file.
It then processes each file within that directory and extracts any
matching e-mails or phone numbers and compares them to the gold file
"""
if __name__ == '__main__':
    if (len(sys.argv) != 3):
        print('usage:\tSpamLord.py <data_dir> <gold_file>')
        sys.exit(0)
    main(sys.argv[1],sys.argv[2])


