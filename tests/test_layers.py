import unittest
import torch
import torch.nn

class Testlayers(unittest.TestCase):

    def utest_embedding_matrix(self):

        input = torch.LongTensor(1, 10).random_(0, 10)
        print(input)
        emb = torch.nn.Embedding(10, 5)

        output = emb(input)
        print(output)

    def utest_nn_dropout(self):

        input = torch.randn(5, 10)
        dropout = torch.nn.Dropout(p=0.3)
        output = dropout(input)
        # print(output)

    def test_enc_conv_block(self):

        input = torch.randn(16, 512, 126).uniform_(-0.1, 0.1)
        print(input.min(), input.max())

        conv = torch.nn.Conv1d(512, 512, 5, padding=2)
        norm = torch.nn.BatchNorm1d(512)
        activation = torch.nn.ReLU()
        dropout = torch.nn.Dropout(p=0.5)

        output = input

        for i in range(3):
            output = conv(output)
            output = norm(output)
            output = activation(output)
            output = dropout(output)

            print(i, "iter")
            print(" max: ", output.max(), "\n min: ", output.min())
            print(" Non zero`s elements:", torch.nonzero(output).size(0), "/", output.nelement())
            print(" Dropout pred: ", 1 - torch.nonzero(output).size(0) / output.nelement())
        print("output shape: ", output.size())


if __name__ == '__main__':
    unittest.main()
