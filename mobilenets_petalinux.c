
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>    // for close()
#include <fcntl.h>     // for open()
#include <sys/mman.h>  // for mmap()


#define IP_BASE 	  0x43C10000 // Mobilenets IP Base Address
#define BRAM_BASE     0x40000000
#define BRAM1_BASE    0x42000000
#define BRAM2_BASE    0x44000000
#define BRAM3_BASE    0x46000000

#define BASE_SIZE 4096       // minimum size to MMAP is 4096

float temp[3072] = {0};
float temp1[27] = {0};
float temp2[9] = {0};
float var;

int main(int argc, char **argv) {
    
    int M = 3072;
    int N = 27;
    int K = 9;
    int O = 2700;
    int r, s, t;
    
    printf("-------------- Starting Test ------------\n");
    
    // Open /dev/mem into file descriptor memfd
    int memfd;
    memfd = open("/dev/mem", O_RDWR | O_SYNC);
    if (memfd == -1) {
        printf("ERROR: memfd - failed to open /dev/mem.\n");
        exit(0);
    }

    // base address of BRAM - DW INPUT:
    off_t base_addr = BRAM_BASE;

    // mmap the base address to BRAM and check
    float *mybram = (float*) mmap(NULL, BASE_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, memfd, base_addr);
    if (mybram == (void*) -1) {
        printf("ERROR: Failed to mmap the BRAM base address \n");
        if (memfd >= 0) 
            close(memfd);        
        exit(0);
    }
    
    FILE *fptr;
    if ((fptr = fopen("image_data.txt","r")) == NULL) 
    {
    	printf ("Error! opening file image_data.txt \n\n");
    	exit (0); // program returns if file is not successfully opened
    }
    
    int m = 0;
    printf("started reading contents of image_data.txt \n\n");
    //while((fgets (mybram[m], 50, (FILE*)fptr)) != NULL) //check for end of file
    while(m<M)
    {

    //    r = fscanf (fptr, "%a", &var); 
	//		temp[m] = var;
	
	      r = fscanf (fptr, "%a", &var); 
			mybram[m] = var;
    		if (m==1)
    		printf ("----------------Values are being written into BRAM------------------\n\n");
    		
    		m++;
    		printf(" written values -> m = %d \n",m);
    		
    	}
    	
    //var = 0;	
    printf("---------Before fclose fptr - BRAM----------\n\n");
    fclose (fptr);
    	
 // printing first 8 values of BRAM
     printf ("----------------printing first 8 values of BRAM------------------\n");
     for (int i=0; i<8; i++) {
        float x = mybram[i];
        // float x = temp[i];
        printf("%d: = %f\n", i, x); // Printing result in hex and decimal
    }
    


/**********************************BRAM1***********************************/
    
    // Open /dev/mem into file descriptor memfd
    int memfd1;
    memfd1 = open("/dev/mem", O_RDWR | O_SYNC);
    if (memfd1 == -1) {
        printf("ERROR: memfd1 - failed to open /dev/mem.\n");
        exit(0);
    }

    // base address of BRAM - DW INPUT:
    off_t base1_addr = BRAM1_BASE;

    // mmap the base address to BRAM and check
    float *mybram1 = (float*) mmap(NULL, BASE_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, memfd1, base1_addr);
    if (mybram1 == (void*) -1) {
        printf("ERROR: mybram1 - Failed to mmap the BRAM base address \n");
        if (memfd1 >= 0) 
            close(memfd1);        
        exit(0);
    }
    
    //FILE *fptr1;
    if ((fptr = fopen("dw_weights_data.txt","r")) == NULL) //SANKAR: Give path to file
    {
    	printf ("Error! opening file dw_weights_data.txt \n");
    	exit (0); // program returns if file is not successfully opened
    }
    
    int n = 0;
    printf("started reading contents of dw_weights_data.txt \n");
   // while((fgets (mybram1[n], 50, (FILE*)fptr)) != NULL) //check for end of file
    while(n<N)
    {

      //s = fscanf (fptr, "%a", &var); 
      //temp1[n] = var;
      
      s = fscanf (fptr, "%a", &var); 
      mybram1[n] = var;
      
    		if (n==1)
    		printf ("----------------Values are being written into BRAM1------------------\n");
    		
    		n++;
    	printf(" written values -> n = %d \n",n);
    }
       printf("---------Before fclose fptr - BRAM1----------\n\n"); 	
    fclose (fptr);
    	
 // printing first 8 values of BRAM
     printf ("----------------printing first 8 values of BRAM1------------------\n");
     for (int i=0; i<8; i++) {
        float x = mybram1[i];
        // float x = temp1[i];
        printf("%d: = %f\n", i, x); // Printing result in hex and decimal
    }
    
    
/**********************************BRAM2***********************************/
    
    // Open /dev/mem into file descriptor memfd
    int memfd2;
    memfd2 = open("/dev/mem", O_RDWR | O_SYNC);
    if (memfd2 == -1) {
        printf("ERROR: memfd2 - failed to open /dev/mem.\n");
        exit(0);
    }

    // base address of BRAM - DW INPUT:
    off_t base2_addr = BRAM2_BASE;

    // mmap the base address to BRAM and check
    float *mybram2 = (float*) mmap(NULL, BASE_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, memfd2, base2_addr);
    if (mybram2 == (void*) -1) {
        printf("ERROR: mybram2 - Failed to mmap the BRAM base address \n");
        if (memfd2 >= 0) 
            close(memfd2);        
        exit(0);
    }
    
    //FILE *fptr;
    if ((fptr = fopen("pw_weights_data.txt","r")) == NULL) //SANKAR: Give path to file
    {
    	printf ("Error! opening file pw_weights_data.txt \n");
    	exit (0); // program returns if file is not successfully opened
    }
    
    int p = 0;
    printf("started reading contents of pw_weights_data \n");
    //while((fgets (mybram2[p], 50, (FILE*)fptr)) != NULL) //check for end of file
    while(p<K)
    {
    
      //t = fscanf (fptr, "%a", &var); 	
      //temp2[p] = var;
      
      t = fscanf (fptr, "%a", &var); 	
      mybram2[p] = var;
      
    		if (p==1)
    		printf ("----------------Values are being written into BRAM2------------------\n");
    		
    		p++;
    		    	printf(" written values -> p = %d \n",p);
    	}
    	
        printf("---------Before fclose fptr - BRAM2----------\n\n");	
    fclose (fptr);
    	
 // printing first 8 values of BRAM
     printf ("----------------printing first 8 values of BRAM2------------------\n");
     for (int i=0; i<8; i++) {
        float x = mybram2[i];
         // float x = temp1[i];
        printf("%d:= %f\n", i, x); // Printing result in hex and decimal
    }

/**********************************Assert start signal for IP***********************************/

   // Open /dev/mem into file descriptor memfd
    int memfd_ip;
    memfd_ip = open("/dev/mem", O_RDWR | O_SYNC);
    if (memfd_ip == -1) {
        printf("ERROR: memfd_ip - failed to open /dev/mem.\n");
        exit(0);
    }

    // base address of BRAM - DW INPUT:
    off_t base_addr_ip = IP_BASE;

    // mmap the base address to BRAM and check
    int *mybramip = (int*) mmap(NULL, BASE_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, memfd_ip, base_addr_ip);
    if (mybramip == (void*) -1) {
        printf("ERROR: mybramip - Failed to mmap the BRAM base address \n");
        if (memfd2 >= 0) 
            close(memfd2);        
        exit(0);
    }
    
    //Asserting start signal
   printf("----------------Asserting start signal------------------- \n");
    mybramip[0] = 1;
    
        while ( (mybramip[1] & 0x1) == 0) {
        ;
    }

    printf("stop encountered = %x \n\r", mybramip[1]);
    
    // Deassert start signal
    mybramip[0] = 0;


/**********************************BRAM3***********************************/
    
    // Open /dev/mem into file descriptor memfd
    int memfd3;
    memfd3 = open("/dev/mem", O_RDWR | O_SYNC);
    if (memfd3 == -1) {
        printf("ERROR: memfd3 - failed to open /dev/mem.\n");
        exit(0);
    }

    // base address of BRAM - DW INPUT:
    off_t base3_addr = BRAM3_BASE;

    // mmap the base address to BRAM and check
    float *mybram3 = (float*) mmap(NULL, BASE_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, memfd3, base3_addr);
    if (mybram3 == (void*) -1) {
        printf("ERROR: mybram3 - Failed to mmap the BRAM base address \n");
        if (memfd3 >= 0) 
            close(memfd3);        
        exit(0);
    }
    
	    	
 // printing output values of BRAM3
     printf ("----------------printing first 8 values of BRAM3------------------\n");
     for (int i=0; i<O; i++) {
        float x = mybram3[i];
        printf("%d: = %f\n", i, x); // Printing result in hex and decimal
    }  
    
    
  
    
// --------------------------- Cleanup --------------------------
     
    // We are done with this mapping, so we unmap it.
    munmap(mybram, BASE_SIZE);

    // We are done with the /dev/mem file descriptor, so we close it.
    close(memfd);
    
// --------------------------- Cleanup --------------------------
        
    // We are done with this mapping, so we unmap it.
    munmap(mybram1, BASE_SIZE);
 

    // We are done with the /dev/mem file descriptor, so we close it.
    close(memfd1);


// --------------------------- Cleanup --------------------------
        
    // We are done with this mapping, so we unmap it.
    munmap(mybram2, BASE_SIZE);
 

    // We are done with the /dev/mem file descriptor, so we close it.
    close(memfd2);

// --------------------------- Cleanup --------------------------
       
    // We are done with this mapping, so we unmap it.
    munmap(mybramip, BASE_SIZE);
 

    // We are done with the /dev/mem file descriptor, so we close it.
    close(memfd_ip);
   
    
    return 0;
}
