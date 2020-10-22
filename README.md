# double\_buffer\_persistent CUDA example

Test implementation of CUDA kernel with double buffering on the output so that the CPU can work on GPU output while the GPU works on the next set of data.
I am curious as to whether this is better than just using CUDA streams to overlap GPU and CPU execution. If kernel launch overhead is high, then maybe this persistent kernel 
with double buffers would beat streams?

Background from Robert Krovella's answer on StackOverflow.com
https://stackoverflow.com/questions/33150040/doubling-buffering-in-cuda-so-the-cpu-can-operate-on-data-produced-by-a-persiste/33158954#33158954

> Here's a fully worked example of a "persistent" kernel, producer-consumer approach, with a double-buffered interface from device (producer) to host (consumer).
> 
> Persistent kernel design generally implies launching kernels with, at most, the number of blocks that can be simultaneously resident on the hardware (see item 1 on slide 16 [here](https://stackoverflow.com/questions/33150040/doubling-buffering-in-cuda-so-the-cpu-can-operate-on-data-produced-by-a-persiste/33158954#33158954)). For the most efficient usage of the machine, we'd generally like to maximize this, while still staying within the aforementioned limit. This involves an occupancy study for a specific kernel, and it will vary from kernel to kernel. Therefore I've chosen to take a shortcut here, and simply launch as many blocks as there are multiprocessors. Such an approach is always guaranteed to work (it could be considered a "lower bound" on the number of blocks to launch for a persistent kernel), but is (typically) not the most efficient usage of the machine. Nevertheless, I claim the occupancy study is beside the point of your question. Furthermore, [it is arguable](https://svail.github.io/persistent_rnns/) that proper "persistent kernel" design with guaranteed forward progress is actually quite tricky - requiring careful design of the CUDA thread code and placement of threadblocks (e.g. only use 1 threadblock per SM) to guarantee forward progress. However we don't need to delve to this level to address your question (I don't think) and the persistent kernel example I propose here only places 1 threadblock per SM.
> 
> I'm also assuming a proper [UVA](http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__UNIFIED.html#group__CUDART__UNIFIED) setup, so that I can skip the details of arranging for proper mapped memory allocations in an non-UVA setup.
> 
> The basic idea is that we will have 2 buffers on the device, along with 2 "mailboxes" in [mapped memory](http://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#zero-copy), one for each buffer. The device kernel will fill a buffer with data, then set the "mailbox" to a value (2, in this case) that indicates the host may "consume" the buffer. The device then goes on to the other buffer and repeats the process in a ping-pong fashion between buffers. In order to make this work we must make sure that the device itself has not overrun the buffers (no thread is allowed to be more than one buffer ahead of any other thread) and that before a buffer is populated by the device, the host has consumed the previous contents.
> 
> On the host side, it is simply waiting for the mailbox to indicate "full", then copying the buffer from device to host, reset the mailbox, and perform the "processing" on it (the validate function). It then goes on to the next buffer in a ping-pong fashion. The actual data "production" by the device is just to fill each buffer with the iteration number. The host then checks to see that the proper iteration number was received.
> 
> I've structured the code to call out the actual device "work" function (my\_compute\_function) which is where you would put whatever your Monte Carlo code is. If your code is nicely thread-independent, this should be straightforward. Thus the device side my\_compute\_function is the producer function, and the host side validate is the consumer function. If your device producer code is not simply thread independent, then you may need to restructure things slightly around the calling point to my\_compute\_function.
> 
> The net effect of this is that the device can "race ahead" and begin filling the next buffer, while the host is "consuming" the data in the previous buffer.
> 
> Because persistent kernel design imposes an upper bound on the number of blocks (and threads) in a kernel launch, I've chosen to implement the "work" producer function in a grid-striding loop, so that arbitrary size buffers can be handled by the given grid-width.
