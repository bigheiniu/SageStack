import pandas as pd
import numpy as np
import xml.etree.cElementTree as et
import pickle
from script.Config import config


def getValueofNode(node):
    return node.text if node is not None else None


def XML2DF(fileName, columnNames):
    parsed_xml = et.parse(fileName)
    df_xml = pd.DataFrame(columns=columnNames)

    for node in parsed_xml.getroot():
        df_xml = df_xml.append(pd.Series([node.attrib.get(col) for col in columnNames], index=columnNames),
                               ignore_index=True)

    return df_xml

def getPostData(fileName):
    # fileName = "/Users/bigheiniu/course/ASU_course/472_social/classproject/stackoverflow/data/Posts.xml"
    columns = ['Id','PostTypeId','Body','ParentId','AcceptedAnswerId','OwnerUserId']
    data = XML2DF(fileName,columns)
    return data


def getVotesRelationship(fileName):
    columns = ['PostId','VoteTypeId']
    data = XML2DF(fileName, columns)
    return data

def saveData(InputFileName, OutputFileName, flag):
    if(flag == 0):
        return
        data = getPostData(InputFileName)
    else:
        data = getVotesRelationship(InputFileName)
    with open(OutputFileName, 'wb') as f1:
        pickle.dump(data, f1)

if __name__ == '__main__':
    input_file_dir = config.ordinary_dir_list
    output_file_dir = config.file_dir_list
    for i in range(len(input_file_dir)):
        saveData(input_file_dir[i], output_file_dir[i],i)
