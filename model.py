from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=512,
            out_channels=1024,
            kernel_size=(1, 3),
            stride=(1, 1)
        )
        self.instance_norm_1 = nn.InstanceNorm1d(num_features=1024)
        self.glu = nn.GLU()
        self.conv2 = nn.Conv1d(
            in_channels=1024,
            out_channels=512,
            kernel_size=(1, 3),
            stride=(1, 1)
        )
        self.instance_norm_2 = nn.InstanceNorm1d(num_features=512)

    def forward(self, input):
        conv1_output = self.conv1(input)
        in1_output = self.instance_norm_1(conv1_output)
        glu1_output = self.glu(in1_output)
        conv2_output = self.conv2(glu1_output)
        in2_output = self.instance_norm_2(conv2_output)
        output = input + in2_output
        return output


class Generator(nn.Module):
    def __init__(self, upscale_factor):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=24,
            out_channels=128,
            kernel_size=(1, 15),
            stride=(1, 1)
        )
        self.glu = nn.GLU()
        self.conv2 = nn.Conv1d(
            in_channels=128,
            out_channels=256,
            kernel_size=(1, 5),
            stride=(1, 2)
        )
        self.instance_norm_1 = nn.InstanceNorm1d(num_features=256)
        self.conv3 = nn.Conv1d(
            in_channels=256,
            out_channels=512,
            kernel_size=(1, 5),
            stride=(1, 2)
        )
        self.instance_norm_2 = nn.InstanceNorm1d(num_features=512)
        self.residual_block = ResidualBlock()
        self.conv4 = nn.Conv1d(
            in_channels=512,
            out_channels=1024,
            kernel_size=(1, 5),
            stride=(1, 1)
        )
        self.pixel_shuffler_1 = nn.PixelShuffle(upscale_factor=upscale_factor)
        self.instance_norm_3 = nn.InstanceNorm1d(num_features=1024)
        self.conv5 = nn.Conv1d(
            in_channels=1024,
            out_channels=512,
            kernel_size=(1, 5),
            stride=(1, 1)
        )
        self.pixel_shuffler_2 = nn.PixelShuffle(upscale_factor=upscale_factor)
        self.instance_norm_4 = nn.InstanceNorm1d(num_features=512)
        self.conv6 = nn.Conv1d(
            in_channels=512,
            out_channels=24,
            kernel_size=(1, 15),
            stride=(1, 1)
        )

    def forward(self, input):
        conv1_output = self.conv1(input)
        glu1_output = self.glu(conv1_output)
        # down sampleing
        conv2_output = self.conv2(glu1_output)
        in1_output = self.instance_norm_1(conv2_output)
        glu2_output = self.glu(in1_output)
        conv3_output = self.conv3(glu2_output)
        in2_output = self.instance_norm_2(conv3_output)
        glu3_output = self.glu(in2_output)
        # residual blocks
        res1_output = self.residual_block(glu3_output)
        res2_output = self.residual_block(res1_output)
        res3_output = self.residual_block(res2_output)
        res4_output = self.residual_block(res3_output)
        res5_output = self.residual_block(res4_output)
        res6_output = self.residual_block(res5_output)
        # up sampling
        conv4_output = self.conv4(res6_output)
        ps1_output = self.pixel_shuffler_1(conv4_output)
        in3_output = self.instance_norm_3(ps1_output)
        glu4_output = self.glu(in3_output)
        conv5_output = self.conv5(glu4_output)
        ps2_output = self.pixel_shuffler_2(conv5_output)
        in4_output = self.instance_norm_4(ps2_output)
        glu5_output = self.glu(in4_output)

        # output
        output = self.conv6(glu5_output)
        return output


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=128,
            kernel_size=(3, 3),
            stride=(1, 2)
        )
        self.glu = nn.GLU()
        self.conv2 = nn.Conv2d(
            in_channels=128,
            out_channels=256,
            kernel_size=(3, 3),
            stride=(2, 2)
        )
        self.instance_norm_1 = nn.InstanceNorm2d(num_features=256)
        self.conv3 = nn.Conv2d(
            in_channels=256,
            out_channels=512,
            kernel_size=(3, 3),
            stride=(2, 2)
        )
        self.instance_norm_2 = nn.InstanceNorm2d(num_features=512)
        self.conv4 = nn.Conv2d(
            in_channels=512,
            out_channels=1024,
            kernel_size=(6, 3),
            stride=(1, 2)
        )
        self.instance_norm_3 = nn.InstanceNorm2d(num_features=1024)
        self.fc = nn.Linear(
            in_features=1024,
            out_features=1
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        conv1_output = self.conv1(input)
        glu1_output = self.glu(conv1_output)
        conv2_output = self.conv2(glu1_output)
        in1_output = self.instance_norm_1(conv2_output)
        glu2_output = self.glu(in1_output)
        conv3_output = self.conv3(glu2_output)
        in2_output = self.instance_norm_2(conv3_output)
        glu3_output = self.glu(in2_output)
        conv4_output = self.conv4(glu3_output)
        in3_output = self.instance_norm_3(conv4_output)
        glu4_output = self.glu(in3_output)
        fc_output = self.fc(glu4_output)
        output = self.sigmoid(fc_output)
        return output
