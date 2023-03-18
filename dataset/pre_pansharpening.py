import os
import numpy as np
import h5py
import cv2
import scipy.io as scio


'''
Datasets download: https://github.com/liangjiandeng/PanCollection
The table of contents is as follows:
--dataset
    |gf2
        |full_examples
        |reduced_examples
        |training_gf2
    |qb
        |full_examples
        |reduced_examples
        |training_qb
    |wv2
        |full_examples
        |reduced_examples
    |wv3
        |full_examples
        |reduced_examples
        |training_wc3
'''

def generate_gf2_train():
    ## gf2 train
    os.makedirs("./train_gf2/gt", exist_ok=True)
    os.makedirs("./train_gf2/lms", exist_ok=True)
    os.makedirs("./train_gf2/ms", exist_ok=True)
    os.makedirs("./train_gf2/pan", exist_ok=True)

    data = h5py.File("./gf2/training_gf2/train_gf2.h5")


    sample_nums = data['gt'][:].shape[0]

    print(sample_nums)
    for i in range(sample_nums):
        gt = data['gt'][:][i, :, :, :]
        f = h5py.File('./train_gf2/gt' + '/' + str(i).zfill(5) + '.h5', 'w')
        f['gt'] = gt


        lms = data['lms'][:][i, :, :, :]
        f = h5py.File('./train_gf2/lms' + '/' + str(i).zfill(5) + '.h5', 'w')
        f['lms'] = lms


        ms = data['ms'][:][i, :, :, :]
        f = h5py.File('./train_gf2/ms' + '/' + str(i).zfill(5) + '.h5', 'w')
        f['ms'] = ms


        pan = data['pan'][:][i, :, :, :]
        f = h5py.File('./train_gf2/pan' + '/' + str(i).zfill(5) + '.h5', 'w')
        f['pan'] = pan



def generate_gf2_test():
    ## gf2 test
    os.makedirs("./test_gf2/test/gt", exist_ok=True)
    os.makedirs("./test_gf2/test/lms", exist_ok=True)
    os.makedirs("./test_gf2/test/ms", exist_ok=True)
    os.makedirs("./test_gf2/test/pan", exist_ok=True)

    data = h5py.File("./gf2/training_gf2/valid_gf2.h5")


    sample_nums = data['gt'][:].shape[0]

    print(sample_nums)
    for i in range(sample_nums):
        gt = data['gt'][:][i, :, :, :]
        f = h5py.File('./test_gf2/test' + '/gt/' + str(i).zfill(5) + '.h5', 'w')
        f['gt'] = gt
        f.close()

        lms = data['lms'][:][i, :, :, :]
        f = h5py.File('./test_gf2/test' + '/lms/' + str(i).zfill(5) + '.h5', 'w')
        f['lms'] = lms
        f.close()

        ms = data['ms'][:][i, :, :, :]
        f = h5py.File('./test_gf2/test' + '/ms/' + str(i).zfill(5) + '.h5', 'w')
        f['ms'] = ms
        f.close()

        pan = data['pan'][:][i, :, :, :]
        f = h5py.File('./test_gf2/test' + '/pan/' + str(i).zfill(5) + '.h5', 'w')
        f['pan'] = pan
        f.close()

    ##gf2 reduced
    os.makedirs("./test_gf2/reduced_examples/gt", exist_ok=True)
    os.makedirs("./test_gf2/reduced_examples/lms", exist_ok=True)
    os.makedirs("./test_gf2/reduced_examples/ms", exist_ok=True)
    os.makedirs("./test_gf2/reduced_examples/pan", exist_ok=True)

    data = h5py.File("./gf2/reduced_examples/test_gf2_multiExm1.h5")


    sample_nums = data['gt'][:].shape[0]

    print(sample_nums)
    for i in range(sample_nums):
        gt = data['gt'][:][i, :, :, :]
        f = h5py.File('./test_gf2/reduced_examples' + '/gt/' + str(i).zfill(5) + '.h5', 'w')
        f['gt'] = gt
        f.close()

        lms = data['lms'][:][i, :, :, :]
        f = h5py.File('./test_gf2/reduced_examples' + '/lms/' + str(i).zfill(5) + '.h5', 'w')
        f['lms'] = lms
        f.close()

        ms = data['ms'][:][i, :, :, :]
        f = h5py.File('./test_gf2/reduced_examples' + '/ms/' + str(i).zfill(5) + '.h5', 'w')
        f['ms'] = ms
        f.close()

        pan = data['pan'][:][i, :, :, :]
        f = h5py.File('./test_gf2/reduced_examples' + '/pan/' + str(i).zfill(5) + '.h5', 'w')
        f['pan'] = pan
        f.close()

    ##gf2 full
    # os.makedirs("./test_gf2/full_examples/gt", exist_ok=True)
    os.makedirs("./test_gf2/full_examples/lms", exist_ok=True)
    os.makedirs("./test_gf2/full_examples/ms", exist_ok=True)
    os.makedirs("./test_gf2/full_examples/pan", exist_ok=True)

    data = h5py.File("./gf2/full_examples/test_gf2_OrigScale_multiExm1.h5")


    sample_nums = data['ms'][:].shape[0]

    print(sample_nums)
    for i in range(sample_nums):
        # gt = data['gt'][:][i, :, :, :]
        # f = h5py.File('./test_gf2/full_examples' + '/' + str(i).zfill(5) + '.h5', 'w')
        # f['gt'] = gt
        # f.close()

        lms = data['lms'][:][i, :, :, :]
        f = h5py.File('./test_gf2/full_examples' + '/lms/' + str(i).zfill(5) + '.h5', 'w')
        f['lms'] = lms
        f.close()

        ms = data['ms'][:][i, :, :, :]
        f = h5py.File('./test_gf2/full_examples' + '/ms/' + str(i).zfill(5) + '.h5', 'w')
        f['ms'] = ms
        f.close()

        pan = data['pan'][:][i, :, :, :]
        f = h5py.File('./test_gf2/full_examples' + '/pan/' + str(i).zfill(5) + '.h5', 'w')
        f['pan'] = pan
        f.close()


def generate_qb_train():
    ## qb train
    os.makedirs("./train_qb/gt", exist_ok=True)
    os.makedirs("./train_qb/lms", exist_ok=True)
    os.makedirs("./train_qb/ms", exist_ok=True)
    os.makedirs("./train_qb/pan", exist_ok=True)

    data = h5py.File("./qb/training_qb/train_qb.h5")


    sample_nums = data['gt'][:].shape[0]

    print(sample_nums)
    for i in range(sample_nums):
        gt = data['gt'][:][i, :, :, :]
        f = h5py.File('./train_qb/gt' + '/' + str(i).zfill(5) + '.h5', 'w')
        f['gt'] = gt
        f.close()

        lms = data['lms'][:][i, :, :, :]
        f = h5py.File('./train_qb/lms' + '/' + str(i).zfill(5) + '.h5', 'w')
        f['lms'] = lms
        f.close()

        ms = data['ms'][:][i, :, :, :]
        f = h5py.File('./train_qb/ms' + '/' + str(i).zfill(5) + '.h5', 'w')
        f['ms'] = ms
        f.close()

        pan = data['pan'][:][i, :, :, :]
        f = h5py.File('./train_qb/pan' + '/' + str(i).zfill(5) + '.h5', 'w')
        f['pan'] = pan
        f.close()


def generate_qb_test():
    ## qb test
    os.makedirs("./test_qb/test/gt", exist_ok=True)
    os.makedirs("./test_qb/test/lms", exist_ok=True)
    os.makedirs("./test_qb/test/ms", exist_ok=True)
    os.makedirs("./test_qb/test/pan", exist_ok=True)

    data = h5py.File("./qb/training_qb/valid_qb.h5")


    sample_nums = data['gt'][:].shape[0]

    print(sample_nums)
    for i in range(sample_nums):
        gt = data['gt'][:][i, :, :, :]
        f = h5py.File('./test_qb/test' + '/gt/' + str(i).zfill(5) + '.h5', 'w')
        f['gt'] = gt
        f.close()

        lms = data['lms'][:][i, :, :, :]
        f = h5py.File('./test_qb/test' + '/lms/' + str(i).zfill(5) + '.h5', 'w')
        f['lms'] = lms
        f.close()

        ms = data['ms'][:][i, :, :, :]
        f = h5py.File('./test_qb/test' + '/ms/' + str(i).zfill(5) + '.h5', 'w')
        f['ms'] = ms
        f.close()

        pan = data['pan'][:][i, :, :, :]
        f = h5py.File('./test_qb/test' + '/pan/' + str(i).zfill(5) + '.h5', 'w')
        f['pan'] = pan
        f.close()

    ##qb reduced
    os.makedirs("./test_qb/reduced_examples/gt", exist_ok=True)
    os.makedirs("./test_qb/reduced_examples/lms", exist_ok=True)
    os.makedirs("./test_qb/reduced_examples/ms", exist_ok=True)
    os.makedirs("./test_qb/reduced_examples/pan", exist_ok=True)

    data = h5py.File("./qb/reduced_examples/test_qb_multiExm1.h5")


    sample_nums = data['gt'][:].shape[0]

    print(sample_nums)
    for i in range(sample_nums):
        gt = data['gt'][:][i, :, :, :]
        f = h5py.File('./test_qb/reduced_examples' + '/gt/' + str(i).zfill(5) + '.h5', 'w')
        f['gt'] = gt
        f.close()

        lms = data['lms'][:][i, :, :, :]
        f = h5py.File('./test_qb/reduced_examples' + '/lms/' + str(i).zfill(5) + '.h5', 'w')
        f['lms'] = lms
        f.close()

        ms = data['ms'][:][i, :, :, :]
        f = h5py.File('./test_qb/reduced_examples' + '/ms/' + str(i).zfill(5) + '.h5', 'w')
        f['ms'] = ms
        f.close()

        pan = data['pan'][:][i, :, :, :]
        f = h5py.File('./test_qb/reduced_examples' + '/pan/' + str(i).zfill(5) + '.h5', 'w')
        f['pan'] = pan
        f.close()

    ##qb full
    # os.makedirs("./test_qb/full_examples/gt", exist_ok=True)
    os.makedirs("./test_qb/full_examples/lms", exist_ok=True)
    os.makedirs("./test_qb/full_examples/ms", exist_ok=True)
    os.makedirs("./test_qb/full_examples/pan", exist_ok=True)

    data = h5py.File("./qb/full_examples/test_qb_OrigScale_multiExm1.h5")


    sample_nums = data['ms'][:].shape[0]

    print(sample_nums)
    for i in range(sample_nums):
        # gt = data['gt'][:][i, :, :, :]
        # f = h5py.File('./test_qb/full_examples' + '/' + str(i).zfill(5) + '.h5', 'w')
        # f['gt'] = gt
        # f.close()

        lms = data['lms'][:][i, :, :, :]
        f = h5py.File('./test_qb/full_examples' + '/lms/' + str(i).zfill(5) + '.h5', 'w')
        f['lms'] = lms
        f.close()

        ms = data['ms'][:][i, :, :, :]
        f = h5py.File('./test_qb/full_examples' + '/ms/' + str(i).zfill(5) + '.h5', 'w')
        f['ms'] = ms
        f.close()

        pan = data['pan'][:][i, :, :, :]
        f = h5py.File('./test_qb/full_examples' + '/pan/' + str(i).zfill(5) + '.h5', 'w')
        f['pan'] = pan
        f.close()


def generate_wv2_test():
    ##wv2 reduced
    os.makedirs("./test_wv2/reduced_examples/gt", exist_ok=True)
    os.makedirs("./test_wv2/reduced_examples/lms", exist_ok=True)
    os.makedirs("./test_wv2/reduced_examples/ms", exist_ok=True)
    os.makedirs("./test_wv2/reduced_examples/pan", exist_ok=True)

    data = h5py.File("./wv2/reduced_examples/test_wv2_multiExm1.h5")

    sample_nums = data['gt'][:].shape[0]

    print(sample_nums)
    for i in range(sample_nums):
        gt = data['gt'][:][i, :, :, :]
        f = h5py.File('./test_wv2/reduced_examples' + '/gt/' + str(i).zfill(5) + '.h5', 'w')
        f['gt'] = gt
        f.close()

        lms = data['lms'][:][i, :, :, :]
        f = h5py.File('./test_wv2/reduced_examples' + '/lms/' + str(i).zfill(5) + '.h5', 'w')
        f['lms'] = lms
        f.close()

        ms = data['ms'][:][i, :, :, :]
        f = h5py.File('./test_wv2/reduced_examples' + '/ms/' + str(i).zfill(5) + '.h5', 'w')
        f['ms'] = ms
        f.close()

        pan = data['pan'][:][i, :, :, :]
        f = h5py.File('./test_wv2/reduced_examples' + '/pan/' + str(i).zfill(5) + '.h5', 'w')
        f['pan'] = pan
        f.close()

    ##wv2 full
    # os.makedirs("./test_wv2/full_examples/gt", exist_ok=True)
    os.makedirs("./test_wv2/full_examples/lms", exist_ok=True)
    os.makedirs("./test_wv2/full_examples/ms", exist_ok=True)
    os.makedirs("./test_wv2/full_examples/pan", exist_ok=True)

    data = h5py.File("./wv2/full_examples/test_wv2_OrigScale_multiExm1.h5")

    sample_nums = data['ms'][:].shape[0]

    print(sample_nums)
    for i in range(sample_nums):
        # gt = data['gt'][:][i, :, :, :]
        # f = h5py.File('./test_wv2/full_examples' + '/' + str(i).zfill(5) + '.h5', 'w')
        # f['gt'] = gt
        # f.close()

        lms = data['lms'][:][i, :, :, :]
        f = h5py.File('./test_wv2/full_examples' + '/lms/' + str(i).zfill(5) + '.h5', 'w')
        f['lms'] = lms
        f.close()

        ms = data['ms'][:][i, :, :, :]
        f = h5py.File('./test_wv2/full_examples' + '/ms/' + str(i).zfill(5) + '.h5', 'w')
        f['ms'] = ms
        f.close()

        pan = data['pan'][:][i, :, :, :]
        f = h5py.File('./test_wv2/full_examples' + '/pan/' + str(i).zfill(5) + '.h5', 'w')
        f['pan'] = pan
        f.close()


def generate_wv3_train():
    ## wv3 train
    os.makedirs("./train_wv3/gt", exist_ok=True)
    os.makedirs("./train_wv3/lms", exist_ok=True)
    os.makedirs("./train_wv3/ms", exist_ok=True)
    os.makedirs("./train_wv3/pan", exist_ok=True)

    data = h5py.File("./wv3/training_wv3/train_wv3.h5")


    sample_nums = data['gt'][:].shape[0]

    print(sample_nums)
    for i in range(sample_nums):
        gt = data['gt'][:][i, :, :, :]
        f = h5py.File('./train_wv3/gt' + '/' + str(i).zfill(5) + '.h5', 'w')
        f['gt'] = gt
        f.close()

        lms = data['lms'][:][i, :, :, :]
        f = h5py.File('./train_wv3/lms' + '/' + str(i).zfill(5) + '.h5', 'w')
        f['lms'] = lms
        f.close()

        ms = data['ms'][:][i, :, :, :]
        f = h5py.File('./train_wv3/ms' + '/' + str(i).zfill(5) + '.h5', 'w')
        f['ms'] = ms
        f.close()

        pan = data['pan'][:][i, :, :, :]
        f = h5py.File('./train_wv3/pan' + '/' + str(i).zfill(5) + '.h5', 'w')
        f['pan'] = pan
        f.close()


def generate_wv3_test():
    ## wv3 test
    os.makedirs("./test_wv3/test/gt", exist_ok=True)
    os.makedirs("./test_wv3/test/lms", exist_ok=True)
    os.makedirs("./test_wv3/test/ms", exist_ok=True)
    os.makedirs("./test_wv3/test/pan", exist_ok=True)

    data = h5py.File("./wv3/training_wv3/valid_wv3.h5")


    sample_nums = data['gt'][:].shape[0]

    print(sample_nums)
    for i in range(sample_nums):
        gt = data['gt'][:][i, :, :, :]
        f = h5py.File('./test_wv3/test' + '/gt/' + str(i).zfill(5) + '.h5', 'w')
        f['gt'] = gt
        f.close()

        lms = data['lms'][:][i, :, :, :]
        f = h5py.File('./test_wv3/test' + '/lms/' + str(i).zfill(5) + '.h5', 'w')
        f['lms'] = lms
        f.close()

        ms = data['ms'][:][i, :, :, :]
        f = h5py.File('./test_wv3/test' + '/ms/' + str(i).zfill(5) + '.h5', 'w')
        f['ms'] = ms
        f.close()

        pan = data['pan'][:][i, :, :, :]
        f = h5py.File('./test_wv3/test' + '/pan/' + str(i).zfill(5) + '.h5', 'w')
        f['pan'] = pan
        f.close()

    ##wv3 reduced
    os.makedirs("./test_wv3/reduced_examples/gt", exist_ok=True)
    os.makedirs("./test_wv3/reduced_examples/lms", exist_ok=True)
    os.makedirs("./test_wv3/reduced_examples/ms", exist_ok=True)
    os.makedirs("./test_wv3/reduced_examples/pan", exist_ok=True)

    data = h5py.File("./wv3/reduced_examples/test_wv3_multiExm1.h5")


    sample_nums = data['gt'][:].shape[0]

    print(sample_nums)
    for i in range(sample_nums):
        gt = data['gt'][:][i, :, :, :]
        f = h5py.File('./test_wv3/reduced_examples' + '/gt/' + str(i).zfill(5) + '.h5', 'w')
        f['gt'] = gt
        f.close()

        lms = data['lms'][:][i, :, :, :]
        f = h5py.File('./test_wv3/reduced_examples' + '/lms/' + str(i).zfill(5) + '.h5', 'w')
        f['lms'] = lms
        f.close()

        ms = data['ms'][:][i, :, :, :]
        f = h5py.File('./test_wv3/reduced_examples' + '/ms/' + str(i).zfill(5) + '.h5', 'w')
        f['ms'] = ms
        f.close()

        pan = data['pan'][:][i, :, :, :]
        f = h5py.File('./test_wv3/reduced_examples' + '/pan/' + str(i).zfill(5) + '.h5', 'w')
        f['pan'] = pan
        f.close()

    ##wv3 full
    # os.makedirs("./test_wv3/full_examples/gt", exist_ok=True)
    os.makedirs("./test_wv3/full_examples/lms", exist_ok=True)
    os.makedirs("./test_wv3/full_examples/ms", exist_ok=True)
    os.makedirs("./test_wv3/full_examples/pan", exist_ok=True)

    data = h5py.File("./wv3/full_examples/test_wv3_OrigScale_multiExm1.h5")


    sample_nums = data['ms'][:].shape[0]

    print(sample_nums)
    for i in range(sample_nums):
        # gt = data['gt'][:][i, :, :, :]
        # f = h5py.File('./test_wv3/full_examples' + '/' + str(i).zfill(5) + '.h5', 'w')
        # f['gt'] = gt
        # f.close()

        lms = data['lms'][:][i, :, :, :]
        f = h5py.File('./test_wv3/full_examples' + '/lms/' + str(i).zfill(5) + '.h5', 'w')
        f['lms'] = lms
        f.close()

        ms = data['ms'][:][i, :, :, :]
        f = h5py.File('./test_wv3/full_examples' + '/ms/' + str(i).zfill(5) + '.h5', 'w')
        f['ms'] = ms
        f.close()

        pan = data['pan'][:][i, :, :, :]
        f = h5py.File('./test_wv3/full_examples' + '/pan/' + str(i).zfill(5) + '.h5', 'w')
        f['pan'] = pan
        f.close()

if __name__ == '__main__':
    # print('--------gf2_train--------')
    # generate_gf2_train()
    # print('--------gf2_test--------')
    # generate_gf2_test()
    # print('--------qb_train--------')
    # generate_qb_train()
    print('--------wv3_train--------')
    generate_wv3_train()

    # print('--------wv3_test--------')
    # generate_wv3_test()
    # print('--------qb_test--------')
    # generate_qb_test()
    # print('--------wc2_test--------')
    # generate_wv2_test()
    # print('--------gf2_test--------')
    # generate_gf2_test()

