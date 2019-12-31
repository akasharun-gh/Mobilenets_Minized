/*   
    Test program for "BRAM Max Value" IP. 

    From "Getting Started with the Xilinx Zynq FPGA and Vivado" 
    by Peter Milder (peter.milder@stonybrook.edu)

    Copyright (C) 2018 Peter Milder

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

#include <stdio.h>
#include "platform.h"
#include "xil_printf.h"

int main() {
    init_platform();

    int randomtests = 1000;

    // Pointers to our BRAM and the control interface of our custom hardware (hw)
    volatile unsigned int* bram = (unsigned int*)XPAR_AXI_BRAM_CTRL_0_S_AXI_BASEADDR;
    volatile unsigned int* bram1 = (unsigned int*)XPAR_AXI_BRAM_CTRL_1_S_AXI_BASEADDR;
    volatile unsigned int* bram2 = (unsigned int*)XPAR_AXI_BRAM_CTRL_2_S_AXI_BASEADDR;
    volatile unsigned int* bram3 = (unsigned int*)XPAR_AXI_BRAM_CTRL_3_S_AXI_BASEADDR;
    volatile unsigned int* hw = (unsigned int*)XPAR_IP_MOBNET_0_S00_AXI_BASEADDR;

    // So, hw[0] will access the first register of our IP, which is ps_control,
    // and hw[1] will access the second register: pl_status
    int M = 3072; //bram1 size
	int N = 27; //bram2 size
	int K = 9; //bram4 size
	int O = 2700;
	//int F = 32; //input image size
	//int L = 30; //output image size
	
	FILE *fptr, *ip_img, *dw_wt, *pw_wt;
	int q, ip, dw, pw;
	float temp, ip_temp, dw_temp, pw_temp;
	
	if((fptr = fopen("out_data.txt","r")) == NULL){
			printf("Error opening file!");
			
			exit(1);
	}
	
	if((ip_img = fopen("float.txt","r")) == NULL){
			printf("Error opening file!");
			
			exit(1);
	}
	
	if((dw_wt = fopen("dw_wt_bram.txt","r")) == NULL){
			printf("Error opening file!");
			
			exit(1);
	}
	
	
	if((pw_wt = fopen("pw_wt_bram.txt","r")) == NULL){
			printf("Error opening file!");
			
			exit(1);
	}

    printf("-------------- Starting Test ------------\n\r");



    xil_printf("Test : Loading data into BRAMs\r\n");

    // The idea here is to make sure that our maxval system is correctly
    // checking *all* 2048 words. So, we can test this by:
    //     - writing the max value 0xffff into the last location in memory
    //     - making sure that none of the other words in memory are that large
	
	int m;

	// Generate random test inputs
	for ( m=0; m<M; m++) {
		ip = fscanf(ip_img, "%b\n" , &ip_temp);
		bram[m] = ip_temp;
	}
	fclose(ip_img);
	// Generate random weights for DC

	int m1;
	for ( m1=0; m1<N; m1++) {
		dw = fscanf(dw_wt, "%b\n" , &dw_temp);
		bram1[m1] = dw_temp;	
	}
	fclose(dw_wt);
				
	//Generate random weights for PC
	int n;
	for ( n=0; n<K; n++){
		pw = fscanf(pw_wt, "%b\n" , &pw_temp);
		bram2[n] = pw_temp;	
	}
	fclose(pw_wt);
	
    // Assert start signal
    hw[0] = 1;

    // Wait for done signal
    while ( (hw[1] & 0x1) == 0) {
        ;
    }

    // Deassert start signal
    hw[0] = 0;
    
	int s1;
	w = 0;
	for( s1=0; s1<O; s1++)
	{
			q = fscanf(fptr, "%f\n" , &temp);
			if(temp != bram3[s1])
			{
			xil_printf("Error: expected %f but got %f at %d\n", bram3[s1], temp, w);
			w++;
			}
	    	//printf("Exp Val: %f  Actual val: %f\n",EO_PW[r2][s1][t5], temp);
	}
	fclose(fptr);
	if(w==0)
	xil_printf("NO ERRORS DETECTED!!:)\n");

    // Now, get ready for test 2. To make sure our IP is ready for a new input,
    // we need to wait until it sets pl_status back to 0:
    while ( (hw[1] & 0x1) != 0) {
        ;
    }


    print("-------------- Done ------------\r\n\n\n\n");

    cleanup_platform();
    return 0;
}
