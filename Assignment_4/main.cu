#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <string.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

#define DATAFILE "./data.bin"
#define OUTFILE "./snapshot.bin"

#define SUPERBLOCK_SIZE 4096 //4KB
#define FCB_SIZE 32768 //32 bytes per FCB * 1024 files = 36864 bytes
#define FCB_ENTRIES 1024
#define STORAGE_SIZE 1085440 //1060KB
#define STORAGE_BLOCK_SIZE 32
#define NAME_LENGTH 20

#define MAX_FILENAME_SIZE 20 //20 bytes
#define MAX_FILE_NUM 1024
#define MAX_FILE_SIZE 1048576 //1024bytes * 1024files = 1048576B = 1024KB

#define G_READ 0
#define G_WRITE 1
#define RM 0  
#define LS_S 1 //list all file by size order
#define LS_D 2 //list all file by modified order

#define U16_NOT_NUMBER 65535 //2^16-1

typedef unsigned char uchar;
typedef uint32_t u32;
typedef uint16_t u16;

typedef struct METADATA {
	int fp;   //starting address
	u16 time; //modified time
	u32 size;
	u16 created_time;
	char fileName[20];
} Meta;

__device__ uchar volume[MAX_FILE_SIZE];
__device__ Meta metadata[FCB_ENTRIES];
__device__ uchar bitmap[SUPERBLOCK_SIZE];
__device__ u16 updated_at = 0;
__device__ u16 created_at = 0;
__device__ u32 file_num = 0;
__device__ u32 last_pos = 0;

__device__ bool isMatched(const char *A, const char *B) {
	int i = -1;
	do {
		i++;
		if (A[i] != B[i]) return false;
	} while (A[i] != 0 && B[i] != 0);
	return true;
}

__device__ bool sizeCmp(const Meta &A, const Meta &B) {
	if (A.size != B.size)
		return A.size > B.size;
	//if with the same size, then first create first print
	return A.created_time < B.created_time;
}

__device__ void sortBySize() {
	for (int i = 0; i < file_num; ++i) {
		for (int j = i + 1; j < file_num; ++j) {
			if (sizeCmp(metadata[j], metadata[i])) {
				// swapping 
				Meta tmp = metadata[i];
				metadata[i] = metadata[j];
				metadata[j] = tmp;
			}
		}
	}
}

__device__ void sortByTime() {
	for (int i = 0; i < file_num; ++i) {
		for (int j = i + 1; j < file_num; ++j) {
			if (metadata[j].time > metadata[i].time) {
				// swapping 
				Meta tmp = metadata[i];
				metadata[i] = metadata[j];
				metadata[j] = tmp;
			}
		}
	}
}

__device__ void sortByCreatedTime() {
	for (int i = 0; i < file_num; ++i) {
		for (int j = i + 1; j < file_num; ++j) {
			if (metadata[j].created_time > metadata[i].created_time) {
				// swapping 
				Meta tmp = metadata[i];
				metadata[i] = metadata[j];
				metadata[j] = tmp;
			}
		}
	}
}

__device__ void sortByFilePointer() {
	for (int i = 0; i < file_num; ++i) {
		for (int j = i + 1; j < file_num; ++j) {
			if (metadata[i].fp > metadata[j].fp) {
				// swapping 
				Meta tmp = metadata[i];
				metadata[i] = metadata[j];
				metadata[j] = tmp;
			}
		}
	}
}

__device__  void shiftFileCreatedTimeCountForOverFlow() {
	sortByCreatedTime();
	created_at = 1;
	for (int i = 0; i< file_num; i++) {
		metadata[i].created_time = created_at;
		created_at++;
	}
}

__device__  void shiftFileTimeCountForOverFlow() {
	sortByTime();
	updated_at= 1;
	for (int i = 0; i< file_num; i++) {
		metadata[i].time = updated_at;
		updated_at++;
	}
}

__device__ u32 open(char *s, int op) {
	u32 fp = 0;
	// Find file is whether exist or not
	for (int i = 0; i < file_num; ++i) 
		if (isMatched(s, metadata[i].fileName)) return i;

	//if the operation is G_WRITE
	if (op) {
		// Create a new file
		created_at++;
		if (file_num == 1024) return (u32)-1; // If more than 1024 files
		metadata[file_num].size = 0;
		metadata[file_num].time = created_at;
		metadata[file_num].created_time = created_at;
		metadata[file_num].fp = last_pos;
		int i = 0;
		do {
			metadata[file_num].fileName[i] = s[i];
		} while (s[i++] != '\0');
		fp = file_num++;
		
		//deal with time overflow
		if (created_at >= U16_NOT_NUMBER) 
			shiftFileCreatedTimeCountForOverFlow();
		return fp;
	}
	return (u32)-1;
}

__device__ void read(uchar *output, u32 size, u32 fp) {
	int i; // a read pointer always start from head of the file
	int j; // a read pointer to identify the location in the file
	 //read bytes data from the file to the output buffer
	for (i = 0, j = metadata[fp].fp; i < size; ++i, ++j) output[i] = volume[j];
}

__device__ void freeSpace() 
{
	//combine fragments to save space
	sortByFilePointer();
	last_pos = 0;
	u32 start_pos, end_pos;
	//restore initial super block
	for (int i = 0; i < SUPERBLOCK_SIZE; ++i) bitmap[i] = 0;
	for (int i = 0; i < file_num; ++i)
	{
		int fp = metadata[i].fp;
		metadata[i].fp = last_pos;

		u32 size = metadata[i].size;
		start_pos = fp;
		end_pos = fp + size;

		if ((end_pos % 32) != 0)
			end_pos = (end_pos / 32) * 32 + 32;

		for (int i = 0; i < 8 - ((start_pos / 32) % 8); ++i)
			bitmap[(start_pos / 256)] |= (1UL << i);

		if ((end_pos / 256) - (start_pos / 256) > 1)
		{
			for (int i = (start_pos / 256); i < (end_pos / 256); ++i)
				for (int j = 0; j<8; j++) bitmap[i] |= (1UL << 7 - j);
		}

		for (int i = 0; i < ((end_pos / 32) % 8); ++i)
			bitmap[(end_pos / 256)] |= (1UL << 7 - i);

		for (int j = 0; j < size; ++j)
			volume[end_pos++] = volume[fp++];

		last_pos = end_pos;
	}
}

__device__ u32 findFreePos(u32 size)
{
	u32 empty_size = 0;
	u32 start_pos;
	int i, j;
	for (i = 0; i <(last_pos / 256); ++i)
	{
		if (bitmap[i] != 255) {
			for (j = 0; j < 8; ++j) {
				if ((bitmap[i] >> 7 - j) & 1U)
				{
					if (empty_size >= size) return start_pos;
					start_pos = (u32)(256 * i + 32 * (j + 1));
					empty_size = 0;
				}
				empty_size += 32;
			}
		}
		else empty_size += 256;
	}
	return (u32)-1;
}

__device__ u32 write(uchar* input, u32 size, u32 fp)
{
	u32 start_pos;
	u32 end_pos;
	//identify the location in the file
	Meta *cur = &metadata[fp];
	u32 free_pos = findFreePos(size);
	if (free_pos != -1)
	{
		cur->fp = free_pos;
		start_pos = end_pos = free_pos;

	} else 
	{
		if (last_pos + size > MAX_FILE_SIZE) freeSpace();
		cur->fp = last_pos;
		start_pos = end_pos = last_pos;
	}

	//updates new info of file
	//cur->fp = last_pos;
	cur->time = updated_at++;
	cur->size = size;

	//take the input buffer to write bytes data to the file
	//u32 current_pos = last_pos;
	for (int i = 0; i < size; ++i)
		volume[end_pos++] = input[i];

	if ((end_pos % 32) != 0)
		end_pos = (end_pos / 32) * 32 + 32;

	//update starting block
	for (int i = 0; i < 8 - ((start_pos / 32) % 8); ++i)
		bitmap[(start_pos / 256)] |= 1UL << i;
	//update middle big blocks
	if ((end_pos / 256) - (start_pos / 256) > 1) 
	{
		for (int i = (start_pos / 256); i < (end_pos / 256); ++i)
			for (int j = 0; j < 8; j++) bitmap[i] |= 1UL << 7 - j;
	}

	//update last big block
	for (int i = 0; i < ((end_pos / 32) % 8); ++i)
		bitmap[(end_pos / 256)] |= 1UL << 7 - i;

	//deal with time overflow
	if (updated_at >= U16_NOT_NUMBER)
		shiftFileTimeCountForOverFlow();

	last_pos = end_pos;
	return last_pos;

}

__device__ void gsys(int op)
{
	if (op == LS_S) {
		sortBySize();
		printf("===sort by file size===\n");
	}
	else if (op == LS_D) {
		sortByTime();
		printf("===sort by modified time===\n");
	}
	//print the info
	for (int i = 0; i < file_num; ++i) {
		Meta *cur = &metadata[i];
		if (op == LS_S)
			printf("%s %d\n", cur->fileName, cur->size);
		else
			printf("%s\n", cur->fileName);
	}
}

__device__ void gsys(int op, char *s)
{	/* Implement remove operation here */
	u32 tar;
	//search the directory for the named file
	for (int i = 0; i < file_num; ++i) {
		if (isMatched(s, metadata[i].fileName)) {
			tar = i;
			break;
		}
	}

	Meta *cur = &metadata[tar];
	u32 start_pos = cur->fp;
	u32 end_pos = start_pos + cur->size;

	if ((end_pos % 32) != 0)
		end_pos = (end_pos / 32) * 32 + 32;
	for (int i = 0; i < 8 - ((start_pos / 32) % 8); ++i)
		bitmap[(start_pos / 256)] |= 0UL << i;
	if ((end_pos / 256) - (start_pos / 256) > 1) {
		for (int i = (start_pos / 256); i < (end_pos / 256); ++i)
			for (int j = 0; j<8; j++) bitmap[i] |= 0UL << 7 - j;
	}
	for (int i = 0; i < ((end_pos / 32) % 8); ++i)
		bitmap[(end_pos / 256)] |= 0UL << 7 - i;

	//delete a file by moving each posterior file forward by 1
	for (int i = tar + 1; i < file_num; ++i)
		metadata[i - 1] = metadata[i];
	file_num--;
}

__global__ void mykernel(uchar *input, uchar *output)
{	//init superblock
	for (int i = 0; i < SUPERBLOCK_SIZE; i++) bitmap[i] = 0;

	// kernel test start
	printf("========KERNEL TEST START=======\n\n");
	//start from input[0], write 64 bytes into t.txt
	u32 fp = open("t.txt\0", G_WRITE);
	write(input, 64, fp);

	//start from input[32], write 32 bytes into b.txt
	fp = open("b.txt\0", G_WRITE);
	write(input + 32, 32, fp);

	//start from input[32], write 32 bytes into t.txt
	fp = open("t.txt\0", G_WRITE);
	write(input + 32, 32, fp);

	//start from beginning of t.txt, read 32bytes and write into output
	fp = open("t.txt\0", G_READ);
	read(output, 32, fp);

	//list all files sort by modified time
	gsys(LS_D);
	//list all files sort by file size
	gsys(LS_S);

	//start from beginning of b.txt, read 64bytes and write into output
	fp = open("b.txt\0", G_WRITE);
	write(input + 64, 12, fp);
	//list all files sort by file size
	gsys(LS_S);
	//list all files sort by modified time
	gsys(LS_D);

	//remove the file t.txt
	gsys(RM, "t.txt\0");
	//list all files sort by file size
	gsys(LS_S);

	// kernel test case 1 end
	printf("========KERNEL TEST 1 END=======\n\n");

	/* test case 2 */
	char fname[10][20];
	for (int i = 0; i < 10; i++) {
		fname[i][0] = i + 33;
		for (int j = 1; j < 19; j++)
			fname[i][j] = 64 + j;
		fname[i][19] = '\0';
	}
	for (int i = 0; i < 10; i++) {
		fp = open(fname[i], G_WRITE);
		write(input + i, 24 + i, fp);
	}
	gsys(LS_S);
	for (int i = 0; i < 5; i++)
		gsys(RM, fname[i]);
	gsys(LS_D);
	//kernel test case 2 end
	printf("========KERNEL TEST 2 END=======\n\n");

	/* test case 3 */
	//kernel test start  
	char fname2[1018][20];
	int p = 0;
	for (int k = 2; k < 15; k++)
		for (int i = 50; i <= 126; i++, p++) {
			fname2[p][0] = i;
			for (int j = 1; j < k; j++)
				fname2[p][j] = 64 + j;
			fname2[p][k] = '\0';
		}
	for (int i = 0; i < 1001; i++) {
		fp = open(fname2[i], G_WRITE);
		write(input + i, 24 + i, fp);
	}
	gsys(LS_S);
	fp = open(fname2[1000], G_READ);
	read(output + 1000, 1024, fp);
	char fname3[17][3];
	for (int i = 0; i < 17; i++) {
		fname3[i][0] = 97 + i;
		fname3[i][1] = 97 + i;
		fname3[i][2] = '\0';
		fp = open(fname3[i], G_WRITE);
		write(input + 1024 * i, 1024, fp);
	}
	fp = open("EA\0", G_WRITE);
	write(input + 1024 * 100, 1024, fp);
	gsys(LS_S);
	//kernel test end
	//kernel test case 3 end
	printf("========KERNEL TEST 3 END=======\n\n");

	/**** kernel test end ****/
	printf("========KERNEL TEST END=======\n");
}

__host__ void write_binaryFile(char *fileName, void *buffer, int bufferSize)
{
	FILE *fp;
	fp = fopen(fileName, "wb");
	fwrite(buffer, 1, bufferSize, fp);
	fclose(fp);
}

__host__ int load_binaryFile(char *fileName, void *buffer, int bufferSize)
{
	FILE *fp;
	fp = fopen(fileName, "rb");

	if (!fp)
	{
		printf("***Unable to open file %s***\n", fileName);
		exit(1);
	}

	//Get file length
	fseek(fp, 0, SEEK_END);
	int fileLen = ftell(fp);
	fseek(fp, 0, SEEK_SET);

	if (fileLen > bufferSize)
	{
		printf("****invalid testcase!!****\n");
		printf("****software warrning: the file: %s size****\n", fileName);
		printf("****is greater than buffer size****\n");
		exit(1);
	}

	//Read file contents into buffer
	fread(buffer, fileLen, 1, fp);
	fclose(fp);
	return fileLen;
}

int main()
{
	uchar *input_h;
	uchar *input;

	uchar *output_h;
	uchar *output;

	input_h = (uchar *)malloc(sizeof(uchar)*MAX_FILE_SIZE);
	output_h = (uchar *)malloc(sizeof(uchar)*MAX_FILE_SIZE);


	cudaMalloc(&input, sizeof(uchar)*MAX_FILE_SIZE);
	cudaMalloc(&output, sizeof(uchar)*MAX_FILE_SIZE);

	// load binary file from data.bin
	load_binaryFile(DATAFILE, input_h, MAX_FILE_SIZE);


	cudaMemcpy(input, input_h, sizeof(uchar)*MAX_FILE_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(output, output_h, sizeof(uchar)*MAX_FILE_SIZE, cudaMemcpyHostToDevice);

	mykernel << <1, 1 >> >(input, output);

	cudaMemcpy(output_h, output, sizeof(uchar)*MAX_FILE_SIZE, cudaMemcpyDeviceToHost);

	// dump output array to snapshot.bin 
	write_binaryFile(OUTFILE, output_h, MAX_FILE_SIZE);

	cudaDeviceSynchronize();
	cudaDeviceReset();

	return 0;
}
