#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <cuda.h>

//page size is 32bytes
#define PAGESIZE 32
//32 KB in shared memory 
#define PHYSICAL_MEM_SIZE 32768
//128 KB in global memory
#define STORAGE_SIZE 131072
//page table entries 1024 (32KB / 32bytes)
#define PT_ENTRIES 1024

#define DATAFILE "./data.bin"
#define OUTFILE "./snapshot.bin"
typedef unsigned char uchar;
typedef uint32_t u32;

//page table entries
__device__ int PAGE_ENTRIES = 0;
//secondary memory
__device__ uchar storage[STORAGE_SIZE];
//pagefault times
__device__ int PAGEFAULT = 0;
//current time
__device__ int inTime = 0;

//page table
extern __shared__ u32 pt[];

__device__ u32 paging(uchar *buffer, u32 frame_num, u32 offset) {
	u32 target;
	int pt_entries = PT_ENTRIES;

	/* The format of entry :
	1. Bit 0 is used to store valid/invalid bit
	2. From bit 1 to 12 is used to store logical page number
	3. From bit 13 to 31 is used to store clock time */

	/* find if there are repeated hitting pages */
	for (int i = 0; i < pt_entries; i++) {
		u32 mask = ((1 << 13) - 2);
		/* pageNum used to store the logic page number of pt[i] */
		u32 pageNum = (pt[i] & mask) >> 1;

		/* If frame_num(the logic page number want to query)
		* is the same as logical page number in entry */
		if ((pt[i] & 1) && pageNum == frame_num) { //page number shall not be zero
			// update hit time  	
			u32 tmpTime = inTime++;
			pt[i] = (tmpTime << 13) | (frame_num << 1) | 1;
			return i * 32 + offset;
		}
	}

	/* shared memory is not FULL */
	for (int i = 0; i < pt_entries; i++) {
		// If find invalid entry (empty entry)
		if ((~pt[i]) & 1) {
			PAGEFAULT++;	// add PageFault
			// update page table and clock time
			u32 tmpTime = inTime++;
			pt[i] = (tmpTime << 13) | (frame_num << 1) | 1;
			return i * 32 + offset;
		}
	}


	/* shared memory is FULL */
	// find least recent used entry
	u32 timeRange = 0; // timeRange = CurrentTime - hitPageTime
	for (int i = 0; i < pt_entries; ++i) {
		u32 mask = (u32)(-1);
		u32 tmpTime = (mask & pt[i]) >> 13;
		u32 tmpTimeRange = inTime - tmpTime;
		//find largest timeRange to determine least recent used entry
		if (tmpTimeRange > timeRange) {
			target = i;	// target variable store the entry 
			timeRange = tmpTimeRange;
		}
	}

	/* move the page from shared memory to global memory
	 * and move the page from secondary storage to shared memory  */
	PAGEFAULT++;
	u32 mask = (1 << 13) - 2;
	u32 tarFrame = (pt[target] & mask) >> 1;	// logical page to be altered
	u32 beginAddress = tarFrame * 32;		// target secondary memory where logical page want to swap in
	for (int i = beginAddress, j = 0; j < 32; i++, j++) {
		u32 sharedAddress = target * 32 + j;    // physical memory address where swapping taking place
		u32 curAddress = frame_num * 32 + j;    // page that are going to swap in physical memory address
		storage[i] = buffer[sharedAddress];		//swap out 
		buffer[sharedAddress] = storage[curAddress];	//swap in 
	}
	int tmpTime = inTime++;
	pt[target] = ((tmpTime) << 13) | (frame_num << 1) | 1;
	return target * 32 + offset;
}

__device__ void init_pageTable(int pt_entries) {

	for (int i = 0; i < pt_entries; i++) {
		pt[i] = 0; //invalid
	}

	for (int i = 0; i < STORAGE_SIZE; i++) {
		storage[i] = 0;
	}

}

__host__ int load_binaryFile(const char *fileName, uchar *buffer, int bufferSize)
{
	FILE *R = fopen(fileName, "rb");
	
	//file open failure
	if (!R)
	{
		printf("***Unable to open file %s***\n", fileName);
		printf("error number is %d\n", errno);
		exit(1);
	}

	//get file length
	fseek(R, 0, SEEK_END);
	int fileLen = ftell(R);
	fseek(R, 0, SEEK_SET);

	if (fileLen > bufferSize)
	{
		printf("****invalid testcase!!****\n");
		printf("****software warrning: the file: %s size****\n", fileName);
		printf("****is greater than buffer size****\n");
		exit(1);
	}

	//read file contents into buffer
	fread(buffer, fileLen, 1, R);
	fclose(R);
	return fileLen;
}

__host__ void write_binaryFile(const char *fileName, uchar *buffer, int bufferSize)
{
	FILE *W = fopen(fileName, "wb");
	fwrite(buffer, 1, bufferSize, W);
	fclose(W);
}

__device__ uchar Gread(uchar *buffer, u32 addr) {
	u32 frame_num = addr / PAGESIZE; // frame num is virtual page number 
	u32 offset = addr % PAGESIZE; // offset is the distance from begining of page
								  //map virtual address to physical address
	addr = paging(buffer, frame_num, offset);
	//read data
	return buffer[addr];
}

__device__ void Gwrite(uchar *buffer, u32 addr, uchar value) {
	u32 frame_num = addr / PAGESIZE; // frame num is virtual page number 
	u32 offset = addr % PAGESIZE; // offset is the distance from begining of page
								  //map virtual address to physical address
	addr = paging(buffer, frame_num, offset);
	//write data
	buffer[addr] = value;
}

__device__ void snapshot(uchar *results, uchar* buffer, int offset, int input_size) {
	for (int i = 0; i < input_size; i++) {
		results[i] = Gread(buffer, i + offset);
	}
}

__global__ void mykernel(int input_size, uchar *input, uchar *results, int *PAGEFAULT_NUM) {

	//take shared memory as physical memory 
	__shared__ uchar data[PHYSICAL_MEM_SIZE];

	//get page table entries 
	int pt_entries = PHYSICAL_MEM_SIZE / PAGESIZE;

	//befor first Gwrite or Gread
	//initialize the page table and entries
	init_pageTable(pt_entries);
	//printf("after init page table\n");
	//printf("[global] Pagefault number is %d\n\n", PAGEFAULT);

	//write data into buffer (physical memory)
	//for (int i = 0; i < 9; i++)
	for (int i = 0; i < input_size; i++)
			Gwrite(data, i, input[i]);
	//printf("after write data into physical memory\n");
	//printf("[global] Pagefault number is %d\n\n", PAGEFAULT);

	//read data from buffer
	for (int i = input_size - 1; i >= input_size - 32769; i--)
		int value = Gread(data, i);
	//printf("after read data from physical memory\n");
	//printf("[global] Pagefault number is %d\n\n", PAGEFAULT);

	//load elements of data array (in shared memory, as physical memeory)
	//to results buffer (in global memory, as secondary storage)
	snapshot(results, data, 0, input_size);

	PAGEFAULT_NUM[0] = PAGEFAULT;
	//printf("after load data to results\n");
	//printf("[global] Pagefault number is %d\n", PAGEFAULT);

}


int main()
{
	cudaError_t cudaStatus;

	uchar *input_h;
	uchar *input;

	uchar *results_h;
	uchar *results;

	int *PAGEFAULT_NUM_h;
	int *PAGEFAULT_NUM;

	cudaMalloc(&input, sizeof(uchar)*STORAGE_SIZE);
	cudaMalloc(&results, sizeof(uchar)*STORAGE_SIZE);
	cudaMalloc(&PAGEFAULT_NUM, sizeof(int));

	input_h = (uchar *)malloc(sizeof(uchar)*STORAGE_SIZE);
	results_h = (uchar *)malloc(sizeof(uchar)*STORAGE_SIZE);
	PAGEFAULT_NUM_h = (int *)malloc(sizeof(int) * 1);
	PAGEFAULT_NUM_h[0] = 0;

	int input_size;

	input_size = load_binaryFile(DATAFILE, input_h, STORAGE_SIZE);
	//printf("[host] Input size is %d\n\n", input_size);

	cudaMemcpy(input, input_h, sizeof(uchar)*STORAGE_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(results, results_h, sizeof(uchar)*STORAGE_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(PAGEFAULT_NUM, PAGEFAULT_NUM_h, sizeof(int) * 1, cudaMemcpyHostToDevice);

	/* Launch kernel function in GPU, with single thread
	and dynamically allocate 16384 bytes of share memory,
	which is used for variables declared as "extern __shared__" */
	mykernel << <1, 1, 16384 >> > (input_size, input, results, PAGEFAULT_NUM);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "mykernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		return 0;
	}

	cudaMemcpy(input_h, input, sizeof(uchar)*STORAGE_SIZE, cudaMemcpyDeviceToHost);
	cudaMemcpy(results_h, results, sizeof(uchar)*STORAGE_SIZE, cudaMemcpyDeviceToHost);
	cudaMemcpy(PAGEFAULT_NUM_h, PAGEFAULT_NUM, sizeof(int), cudaMemcpyDeviceToHost);

	//dump the contents of binary file into snapshot.bin
	write_binaryFile(OUTFILE, results_h, input_size);

	printf("[host] Pagefault number is %d\n", PAGEFAULT_NUM_h[0]);

	cudaFree(input);
	cudaFree(results);
	cudaFree(PAGEFAULT_NUM);

	free(input_h);
	free(results_h);
	free(PAGEFAULT_NUM_h);

	cudaDeviceSynchronize();
	cudaDeviceReset();

	return 0;
}
