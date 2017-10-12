# coding: utf-8
import os


# calculate the word count in document
def ReadFile(path, start_index):
    word_counter = {}
    document_info = []
    document_word_count = 0

    for i, line in enumerate(open(path, 'r')):
        word_list = line.replace('\n', '').replace('-1', '').strip()

        if (i < start_index):
            document_info.append(word_list)
        else:
            for word in word_list.split():
                temp = str(word)
                document_word_count += 1
                if temp not in word_counter:
                    word_counter[temp] = 1
                else:
                    word_counter[temp] += 1
    return document_info, word_counter, document_word_count


# debug
def ReadFileDebug(document_info, word_counter):
    for info in document_info:
        print(info)
    print('-' * 18)
    print('{:15}{:3}'.format('Word', 'Count'))
    print('-' * 18)
    for (word, occurance) in word_counter.items():
        print('{:15}{:3}'.format(word, occurance))


def ReadFolder(p, start_index):
    folder = {}
    files = os.listdir(p)
    files_name = []
    folder_word_counter = {}
    folder_word_counter_distince = {}  # word出現在各個document的次數（如果出現為1）

    for f in files:  # f is file name
        fullpath = os.path.join(p, f)
        if os.path.isfile(fullpath):
            files_name.append(f)

    # read all files in the folder
    for fn in files_name:
        fullpath = os.path.join(p, fn)
        if (fn == '.DS_Store'): continue
        document_info, word_counter, document_word_count = ReadFile(fullpath, start_index)

        folder[fn] = word_counter
        # debug
        # if debug: ReadFileDebug(document_info, word_counter)

    # files_count = len(folder)

    # calculate all word count in the folder
    for (f, words) in folder.items():
        for (word, count) in words.items():
            if word not in folder_word_counter:
                folder_word_counter[word] = count
                folder_word_counter_distince[word] = 1
            else:
                folder_word_counter[word] += count
                folder_word_counter_distince[word] += 1

    # doc word count/ folder word count/ folder word distinct count
    return folder, folder_word_counter, folder_word_counter_distince


def ReadFolderDebug(po, doc_word_count, folder_word_count, folder_word_count_distinct):
    print('{:30}{:15}{:3}'.format('File Name', 'Word', 'Count'))
    print('-' * 20)
    for (fn, words) in doc_word_count.items():
        # if (os.path.exists(os.path.join(po, fn))):
        #     os.remove(os.path.join(po, fn))
        # f = open(os.path.join(po, fn), 'w')
        for (word, count) in words.items():
            #f.writelines(str(word) + ": " + str(count) + '\n')
            print('{:30}{:15}{:3}'.format(fn, word, count))
        #f.close()


def ReadBGLMFile(path):
    dict = {}

    for i, line in enumerate(open(path, 'r')):
        word_list = line.replace('\n', '').strip()
        ws = word_list.split()
        dict[int(ws[0])] = float(ws[1])

    return dict


def ReadCollectionFile(path):
    docs_dict = {}
    index = 0

    for i, line in enumerate(open(path, 'r')):
        word_list = line.replace('\n', '').strip()
        ws = word_list.split()
        # print(ws)
        words_dict = {}
        for word in ws:
            temp = int(word)
            if temp not in words_dict:
                words_dict[temp] = 1
            else:
                words_dict[temp] += 1

        docs_dict[index] = words_dict
        #print(len(ws), sum(words_dict.values()))
        index += 1

    return docs_dict


def ReadCollectionFileDebug(docs_dict):
    for file, words in docs_dict.items():
        for word, count in words.items():
            print(file, word, count)


if __name__ == '__main__':
    debug1 = False
    debug2 = False
    debug3 = False
    debug4 = False
    debug5 = False

    p = '../dataset/Query/3001.query'
    pd = '../dataset/Document'
    po = '../dataset/Output'
    pq = '../dataset/Query'
    pbglm = '../dataset/BGLM.txt'
    pcollection = '../dataset/Collection.txt'
    start_index = 0

    # file
    document_info, word_counter, document_word_count = ReadFile(p, start_index)
    # debug function
    if debug1: ReadFileDebug(document_info, word_counter)

    # documents
    doc_word_count, folder_word_count, folder_word_count_distinct = ReadFolder(pd, start_index)
    # debug function
    if debug2: ReadFolderDebug(po, doc_word_count, folder_word_count, folder_word_count_distinct)

    # query
    start_index = 0
    doc_word_count, folder_word_count, folder_word_count_distinct = ReadFolder(pq, start_index)
    if debug3: ReadFolderDebug(po, doc_word_count, folder_word_count, folder_word_count_distinct)

    # BGLM
    bglm = ReadBGLMFile(pbglm)
    # debug
    if debug5:
        for word, count in bglm.items():
            if (count < 0): print(word, count)

    collection = ReadCollectionFile(pcollection)
    if debug4: ReadCollectionFileDebug(collection)
