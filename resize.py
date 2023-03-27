from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pyopencl as cl


class CResize(object):
    def __init__(self):
        platforms = cl.get_platforms()
        print("Available devices:")
        for platform in platforms:
            devices = platform.get_devices()
            for device in devices:
                print(
                    "  * {0} on {1} ({2})".format(
                        device.name, device.platform.name, device.platform.version
                    )
                )

        device = platforms[0].get_devices()[0]
        print(
            "Picked {0} on {1} ({2})".format(
                device.name, device.platform.name, device.platform.version
            )
        )

        self.ctx = cl.Context([device])
        self.queue = cl.CommandQueue(self.ctx)

        print(
            "Max allowed 2D image size: ({0}, {1})".format(
                device.get_info(cl.device_info.IMAGE2D_MAX_WIDTH),
                device.get_info(cl.device_info.IMAGE2D_MAX_HEIGHT),
            )
        )
        print(
            "Max allowed 3D image size: ({0}, {1}, {2})".format(
                device.get_info(cl.device_info.IMAGE3D_MAX_WIDTH),
                device.get_info(cl.device_info.IMAGE3D_MAX_HEIGHT),
                device.get_info(cl.device_info.IMAGE3D_MAX_DEPTH),
            )
        )

        with open("./kernels/resize.cl", "r", encoding="utf-8") as f:
            resize_kernel_str = f.read().strip()
            self.prg_resize = cl.Program(self.ctx, resize_kernel_str).build()

        self.channel_mapping = {
            1: cl.channel_order.R,
            2: cl.channel_order.RG,
            3: cl.channel_order.RGB,
            4: cl.channel_order.RGBA,
        }

    def get_num_channels_and_order(self, src_data, physical_dim):
        num_channels = 1
        if src_data.ndim < physical_dim:
            raise ValueError(
                str(physical_dim)
                + "D image must be of dimension "
                + str(physical_dim)
                + " if only one channel, or "
                + str(physical_dim + 1)
                + ", but was "
                + str(src_data.ndim)
            )
        elif src_data.ndim == physical_dim + 1:
            num_channels = src_data.shape[physical_dim]
            if num_channels > 4:
                raise ValueError(
                    "Invalid number of channels ("
                    + str(num_channels)
                    + "), must be 1, 2, 3 or 4"
                )

        return num_channels, self.channel_mapping[num_channels]

    def get_channel_type(self, src_data):
        if src_data.dtype == np.uint8:
            channel_type = cl.channel_type.UNSIGNED_INT8
        elif src_data.dtype == np.uint16:
            channel_type = cl.channel_type.UNSIGNED_INT16
        elif src_data.dtype == np.uint32:
            channel_type = cl.channel_type.UNSIGNED_INT32
        elif src_data.dtype == np.float32:
            channel_type = cl.channel_type.FLOAT
        else:
            raise TypeError(
                str("Data type " + str(src_data.dtype) + " currently not supported")
            )
        return channel_type

    def resize2D(self, src_data, target_shape, mode="NEAREST"):
        if len(target_shape) != 2:
            raise ValueError(
                "Invalid target shape dimension ("
                + str(len(target_shape))
                + "). Should be 2-dimensional"
            )
        if any(map(lambda x: x < 1, target_shape)):
            raise ValueError("Invalid target shape: " + str(target_shape))

        num_channels, channel_order = self.get_num_channels_and_order(src_data, 2)
        channel_type = self.get_channel_type(src_data)

        fmt = cl.ImageFormat(channel_order, channel_type)
        src_img = cl.image_from_array(self.ctx, src_data, num_channels=num_channels)
        dst_img = cl.Image(
            self.ctx, cl.mem_flags.WRITE_ONLY, fmt, target_shape[::-1]
        )  # img dims are swapped

        roi_shape = tuple(np.maximum(src_img.shape, dst_img.shape))

        if mode == "NEAREST":
            self.prg_resize.nearest2D(self.queue, roi_shape, None, src_img, dst_img)
        elif mode == "LINEAR":
            self.prg_resize.linear2D(self.queue, roi_shape, None, src_img, dst_img)

        res = np.empty(target_shape + (num_channels,), src_data.dtype)
        cl.enqueue_copy(self.queue, res, dst_img, origin=(0, 0), region=dst_img.shape)

        return res

    def resize3D(self, src_data, target_shape, mode="NEAREST"):
        if len(target_shape) != 3:
            raise ValueError(
                "Invalid target shape dimension ("
                + str(len(target_shape))
                + "). Should be 3-dimensional"
            )
        if any(map(lambda x: x < 1, target_shape)):
            raise ValueError("Invalid target shape: " + str(target_shape))

        num_channels, channel_order = self.get_num_channels_and_order(src_data, 3)
        channel_type = self.get_channel_type(src_data)

        fmt = cl.ImageFormat(channel_order, channel_type)
        src_img = cl.image_from_array(self.ctx, src_data, num_channels=num_channels)
        dst_img = cl.Image(
            self.ctx, cl.mem_flags.WRITE_ONLY, fmt, target_shape[::-1]
        )  # img dims are swapped

        roi_shape = tuple(np.maximum(src_img.shape, dst_img.shape))

        if mode == "NEAREST":
            self.prg_resize.nearest3D(self.queue, roi_shape, None, src_img, dst_img)
        elif mode == "LINEAR":
            self.prg_resize.linear3D(self.queue, roi_shape, None, src_img, dst_img)

        res = np.empty(target_shape + (num_channels,), src_data.dtype)
        cl.enqueue_copy(
            self.queue, res, dst_img, origin=(0, 0, 0), region=dst_img.shape
        )

        return res


def main():
    # Init OpenCL (only once)
    resizer = CResize()

    # Read picture data
    src_data = np.ascontiguousarray(
        np.array(Image.open("./images/cat.jpg").convert("RGBA"), dtype=np.float32)
        / 255.0
    )
    target_shape = (src_data.shape[0] // 2, src_data.shape[1] // 2)

    # Resize picture
    res_nearest = resizer.resize2D(src_data, target_shape, mode="NEAREST")
    res_linear = resizer.resize2D(src_data, target_shape, mode="LINEAR")

    # Show algorithm results
    plt.figure(figsize=(10, 6), dpi=97)
    plt.subplot(121)
    plt.imshow(res_nearest)
    plt.title("Nearest Resize Algorithm")
    plt.subplot(122)
    plt.imshow(res_linear)
    plt.title("Linear Resize Algorithm")
    plt.show()


if __name__ == "__main__":
    main()
