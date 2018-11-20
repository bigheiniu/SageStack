import pandas as pd
import numpy as np
import xml.etree.cElementTree as et
import pickle
from script.Config import config
import os


def getValueofNode(node):
    return node.text if node is not None else None


def XML2DF(fileName, columnNames,flag="Vote"):
    parsed_xml = et.parse(fileName)

    store = {col:[] for col in columnNames}
    for node in parsed_xml.getroot():
        for col in columnNames:
            store.get(col).append(node.attrib.get(col))
    df_xml = pd.DataFrame(store)
    if flag == "Post":
        content = []
        for node in parsed_xml.getroot():
            content.append(node.attrib.get("Body"))
        return df_xml, content
    return df_xml

def getPostData(fileName):
    # fileName = "/Users/bigheiniu/course/ASU_course/472_social/classproject/stackoverflow/data/Posts.xml"
    columns = ['Id','PostTypeId','ParentId','AcceptedAnswerId','OwnerUserId']
    data,content = XML2DF(fileName,columns,"Post")
    return data,content


def getVotesRelationship(fileName):
    columns = ['PostId','VoteTypeId']
    data = XML2DF(fileName, columns)
    return data

def saveData(InputFileName, OutputFileName, flag):
    if(flag == 0):
        data, content = getPostData(InputFileName)
        with open(config.resource_base_dir + "content_list.pickle", "wb") as f1:
            pickle.dump(content, f1)
    else:
        data = getVotesRelationship(InputFileName)
    with open(OutputFileName, 'wb') as f1:
        pickle.dump(data, f1)

if __name__ == '__main__':
    input_file_dir = config.ordinary_dir_list
    output_file_dir = config.file_dir_list
    for i in range(len(input_file_dir)):
        saveData(input_file_dir[i], output_file_dir[i],i)
