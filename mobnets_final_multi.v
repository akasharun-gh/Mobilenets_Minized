module top (
    parameter T=32;
    input clk,
    input reset,
    input [T-1:0] start, next_set,
    input [T-1:0] bram1_rddata,
    input [T-1:0] bram2_rddata,
    input [T-1:0] bram4_rddata,
    input [T-1:0] bram5_rddata,
    output   [T-1:0] bram1_addr,   // 32*32*3 = 3072 * 4 = 12288 bytes
    output   [T-1:0]  bram2_addr,  // 3*3 = 9 *3 = 27(kernels)*4 = 108 bytes
    output   [T-1:0]  bram4_addr,  // 3*3 = 9 *4 = 36 bytes
    output [T-1:0] bram5_addr,  // 30*30*3 = 2700 * 4 = 10,800 bytes
    output [T-1:0] bram1_wrdata,
    output [T-1:0] bram2_wrdata,
    output [T-1:0] bram4_wrdata,
    output      [T-1:0] bram5_wrdata,
    output      [T-1:0] stop,
    output      [3:0]  bram5_we);
    
    wire bram3_we, bram3_re;
    wire [T-1:0] bram3_rddata;
    wire       [12:0] bram3_addr_rd; //2700 words of 66 bits each
    wire        [12:0] bram3_addr_wr;
    wire    [T-1:0] bram3_wrdata;
    wire    [T-1:0] acc_reg;
    wire    [T-1:0] temp;
    wire    [T-1:0] m_axis_result_tdata;
    wire    [T-1:0] pw_m_axis_result_tdata;
    wire        s_axis_a_tvalid;
    wire        s_axis_b_tvalid;
    wire        s_axis_c_tvalid;
    wire        m_axis_result_tvalid;
    wire        pw_s_axis_a_tvalid;
    wire        pw_s_axis_b_tvalid;
    wire        pw_s_axis_c_tvalid;
    wire        pw_m_axis_result_tvalid;

    depthwise #(4,14) dw  (.clk(clk), .reset(reset), .bram1_rddata(bram1_rddata), .bram2_rddata(bram2_rddata), .bram3_we(bram3_we), .bram3_re(bram3_re),
    .bram1_addr(bram1_addr), .bram2_addr(bram2_addr), .bram3_addr_wr(bram3_addr_wr), .bram3_wrdata(bram3_wrdata), .start(start),
    .m_axis_result_tvalid(m_axis_result_tvalid), .m_axis_result_tdata(m_axis_result_tdata), .s_axis_a_tvalid(s_axis_a_tvalid),
    .s_axis_b_tvalid(s_axis_b_tvalid), .s_axis_c_tvalid(s_axis_c_tvalid), .acc_reg(acc_reg), .next_set(next_set));
    
    pointwise #(4,14) pw(.clk(clk), .reset(reset), .bram3_rddata(bram3_rddata), .bram4_rddata(bram4_rddata), .bram5_we(bram5_we), .bram3_re(bram3_re),
                .bram3_we(bram3_we), .bram4_addr(bram4_addr), .bram3_addr_rd(bram3_addr_rd), .bram5_addr(bram5_addr), .bram5_wrdata(bram5_wrdata),
                .start(start), .stop(stop), .pw_s_axis_a_tvalid(pw_s_axis_a_tvalid), .pw_s_axis_b_tvalid(pw_s_axis_b_tvalid), .pw_s_axis_c_tvalid(pw_s_axis_c_tvalid),
                .pw_m_axis_result_tvalid(pw_m_axis_result_tvalid), .pw_m_axis_result_tdata(pw_m_axis_result_tdata), .temp(temp), .next_set(next_set));
    
    bram3 bram  (.clk(clk), .reset(reset), .bram3_we(bram3_we), .bram3_re(bram3_re), .bram3_wrdata(bram3_wrdata),
    .bram3_addr_wr(bram3_addr_wr),  .bram3_addr_rd(bram3_addr_rd), .bram3_rddata(bram3_rddata));
    
    floating_point_0 dwfma (
                       .aclk(clk),                                  // input wire aclk
                       .s_axis_a_tvalid(s_axis_a_tvalid),            // input wire s_axis_a_tvalid
                       .s_axis_a_tdata(bram1_rddata),              // input wire [31 : 0] s_axis_a_tdata
                       .s_axis_b_tvalid(s_axis_b_tvalid),            // input wire s_axis_b_tvalid
                       .s_axis_b_tdata(bram2_rddata),              // input wire [31 : 0] s_axis_b_tdata
                       .s_axis_c_tvalid(s_axis_c_tvalid),            // input wire s_axis_c_tvalid
                       .s_axis_c_tdata(acc_reg),              // input wire [31 : 0] s_axis_c_tdata
                       .m_axis_result_tvalid(m_axis_result_tvalid),  // output wire m_axis_result_tvalid
                       .m_axis_result_tdata(m_axis_result_tdata)    // output wire [31 : 0] m_axis_result_tdata
                     );
         
    floating_point_0 pwfma (
                        .aclk(clk),                                  // input wire aclk
                        .s_axis_a_tvalid(pw_s_axis_a_tvalid),            // input wire s_axis_a_tvalid
                        .s_axis_a_tdata(bram3_rddata),              // input wire [31 : 0] s_axis_a_tdata
                        .s_axis_b_tvalid(pw_s_axis_b_tvalid),            // input wire s_axis_b_tvalid
                        .s_axis_b_tdata(bram4_rddata),              // input wire [31 : 0] s_axis_b_tdata
                        .s_axis_c_tvalid(pw_s_axis_c_tvalid),            // input wire s_axis_c_tvalid
                        .s_axis_c_tdata(temp),              // input wire [31 : 0] s_axis_c_tdata
                        .m_axis_result_tvalid(pw_m_axis_result_tvalid),  // output wire m_axis_result_tvalid
                        .m_axis_result_tdata(pw_m_axis_result_tdata)    // output wire [31 : 0] m_axis_result_tdata
                      );
                             
endmodule

module depthwise(
    parameter T=32;
    input clk,
    input reset,
    input m_axis_result_tvalid,
    input [T-1:0]  start, next_set,
    input [T-1:0] bram1_rddata,
    input [T-1:0] bram2_rddata,
    input       [T-1:0] m_axis_result_tdata,
    output reg   bram3_we,
    output reg   bram3_re,
    output reg [T-1:0] bram1_addr,  
    output reg [T-1:0]  bram2_addr,    
    output reg  [12:0] bram3_addr_wr, // 30*30*3 = 2700 values of 66 bits each
    output reg  [T-1:0] bram3_wrdata,
    output reg  [T-1:0] acc_reg,
    output reg  s_axis_a_tvalid, //SANKAR
    output reg  s_axis_b_tvalid, //SANKAR
    output reg  s_axis_c_tvalid);//SANKAR
    parameter ADDR_JUMP = 1;
    parameter LATENCY = 1;
    
    reg [2:0]  cnt; // input feature map counter needed for control logic  
    reg [5:0]  wt_cnt; // weight counter iterating through depthwise weights BRAM
    reg [3:0]  acc_cnt;
    reg [3 : 0] latency; //= LATENCY;
    // signals required for addressing the input feature map bram correctly
    reg [T-1:0] val1_addr, val2_addr, val3_addr, val1_g_addr, val2_g_addr, val3_g_addr, val1_b_addr, val2_b_addr, val3_b_addr ;
    reg inc1, inc2, inc3;
    reg [4:0] col_cnt;
    reg flag, flag1, r_en, g_en, b_en;
    reg [T-1:0] acc_temp;
    



    //latency control
    always@(posedge clk) begin
    if(!reset)
        latency<=0;
    else if(next_set)
        latency <= 0;
    else if (start)begin
        if (latency == LATENCY)
            latency <= 0;
         else
        latency<= latency+1;
        end
    end
    // control logic block for addressing the input feature map bram
    always @(posedge clk) begin
    if(!reset)
    begin
        bram1_addr <= 0;
        r_en <= 1;
        g_en <= 0;
        b_en <= 0;
        val1_addr <= 0;
        val2_addr <= 31*ADDR_JUMP;
        val3_addr <= 63*ADDR_JUMP;
        val1_g_addr <= 1023*ADDR_JUMP;
        val2_g_addr <= 1055*ADDR_JUMP;
        val3_g_addr <= 1087*ADDR_JUMP;
        val1_b_addr <= 2047*ADDR_JUMP;
        val2_b_addr <= 2079*ADDR_JUMP;
        val3_b_addr <= 2111*ADDR_JUMP;
    end
    else if (next_set) begin
        bram1_addr <= 0;
        r_en <= 1;
        g_en <= 0;
        b_en <= 0;
        val1_addr <= 0;
        val2_addr <= 31*ADDR_JUMP;
        val3_addr <= 63*ADDR_JUMP;
        val1_g_addr <= 1023*ADDR_JUMP;
        val2_g_addr <= 1055*ADDR_JUMP;
        val3_g_addr <= 1087*ADDR_JUMP;
        val1_b_addr <= 2047*ADDR_JUMP;
        val2_b_addr <= 2079*ADDR_JUMP;
        val3_b_addr <= 2111*ADDR_JUMP;
    end
    else
    begin
    
    if( acc_cnt!=9 && start && !latency && !flag1/*== 32'h0000ffff*/) begin
        if(r_en)begin //R feature map
            if(inc1)
            begin
            if(cnt == 0) begin
            bram1_addr <= val1_addr + ADDR_JUMP;
            if(col_cnt == 28)
            val1_addr <= val1_addr + ADDR_JUMP*3;
            else
            val1_addr <= val1_addr + ADDR_JUMP;
            end
        else
        bram1_addr <= bram1_addr + ADDR_JUMP;
    end
    
    else if(inc2)
    begin
        if(cnt == 0) begin
        bram1_addr <= val2_addr + ADDR_JUMP;
        if(col_cnt == 29)
        val2_addr <= val2_addr + ADDR_JUMP*3;
        else
        val2_addr <= val2_addr + ADDR_JUMP;
        end
        else
        bram1_addr <= bram1_addr + ADDR_JUMP;
    end
    
    else if(inc3)
    begin
    if(cnt == 0) begin
    bram1_addr <= val3_addr + ADDR_JUMP;
    if(col_cnt == 29)
    val3_addr <= val3_addr + ADDR_JUMP*3;
    else
    val3_addr <= val3_addr + ADDR_JUMP;
    end
    else
    bram1_addr <= bram1_addr + ADDR_JUMP;
    if(cnt == 2) begin
    g_en <= 1;
    r_en <= 0;
    end
    end
    end
    
    if(g_en) begin //G feature map
    if(inc1)
    begin
    if(cnt == 0) begin
    bram1_addr <= val1_g_addr + ADDR_JUMP;
    if(col_cnt == 29)
    val1_g_addr <= val1_g_addr + ADDR_JUMP*3;
    else
    val1_g_addr <= val1_g_addr + ADDR_JUMP;
    end
    else
    bram1_addr <= bram1_addr + ADDR_JUMP;
    end
    
    else if(inc2)
    begin
    if(cnt == 0) begin
    bram1_addr <= val2_g_addr + ADDR_JUMP;
    if(col_cnt == 29)
    val2_g_addr <= val2_g_addr + ADDR_JUMP*3;
    else
    val2_g_addr <= val2_g_addr + ADDR_JUMP;
    end
    else
    bram1_addr <= bram1_addr + ADDR_JUMP;
    end
    
    else if(inc3)
    begin
    if(cnt == 0) begin
    bram1_addr <= val3_g_addr + ADDR_JUMP;
    if(col_cnt == 29)
    val3_g_addr <= val3_g_addr + ADDR_JUMP*3;
    else
    val3_g_addr <= val3_g_addr + ADDR_JUMP;
    end
    else
    bram1_addr <= bram1_addr + ADDR_JUMP;
    if(cnt == 2) begin
    b_en <= 1;
    g_en <= 0;
    end
    end
    end
    
    if(b_en) begin //B feature map
    if(inc1)
    begin
    if(cnt == 0) begin
    bram1_addr <= val1_b_addr + ADDR_JUMP;
    if(col_cnt == 29)
    val1_b_addr <= val1_b_addr + ADDR_JUMP*3;
    else
    val1_b_addr <= val1_b_addr + ADDR_JUMP;
    end
    else
    bram1_addr <= bram1_addr + ADDR_JUMP;
    end
    
    else if(inc2)
    begin
    if(cnt == 0) begin
    bram1_addr <= val2_b_addr + ADDR_JUMP;
    if(col_cnt == 29)
    val2_b_addr <= val2_b_addr + ADDR_JUMP*3;
    else
    val2_b_addr <= val2_b_addr + ADDR_JUMP;
    end
    else
    bram1_addr <= bram1_addr + ADDR_JUMP;
    end
    
    else if(inc3)
    begin
    if(cnt == 0) begin
    bram1_addr <= val3_b_addr + ADDR_JUMP;
    if(col_cnt == 29)
    val3_b_addr <= val3_b_addr + ADDR_JUMP*3;
    else
    val3_b_addr <= val3_b_addr + ADDR_JUMP;
    end
    else
    bram1_addr <= bram1_addr + ADDR_JUMP;
    if(cnt == 2) begin
    r_en <= 1;
    b_en <= 0;
    end
    end
    end
    
    end // end of if(start)
    end // end of reset else
    
    end // end of posedge block


// block for resetting the counter every 3 cycles for reading from input feature map bram
always@ (posedge clk) begin

if(!reset)
cnt <= 1;
else if(next_set)
    cnt <= 1;
else
begin
    if(acc_cnt!=9 && start && !latency /*== 32'h0000ffff*/) begin
        if(cnt == 2)
        cnt <= 0;
        else
        cnt <= cnt + 1;
        end
    end
end

// Boundary condition check - 2nd row
always@ (posedge clk) begin

    if(!reset)
    col_cnt <= 0;
    else begin
    if(next_set)
        col_cnt <= 0;
    else begin
    if(acc_cnt!=9 && start && wt_cnt == 27 && !latency) begin
    if(col_cnt == 29)
    col_cnt <= 0;
    else
    col_cnt <= col_cnt + 1;
    end
    end
    end
end

// block for updating the values of registers and wires for reading input feature map from bram
// There are a few hardcoded values in this block required for addressing input maps correctly
always @(posedge clk) begin

if(!reset)
begin
inc1 <= 0;
inc2 <= 0;
inc3 <= 0;

end
else
begin
if(next_set) begin
    inc1 <= 0;
    inc2 <= 0;
    inc3 <= 0;
end
else begin
if(bram1_addr == 0)  
inc1 <= 1;
if(acc_cnt!=9 && cnt == 2 && start && !latency)
begin
inc1 <= inc3;
inc2 <= inc1;
inc3 <= inc2;

end
end // end of next_set
end // end of reset else

end // end of block

// control logic required for addressing depthwise weight bram
always@(posedge clk) begin
if (!reset) begin
bram2_addr <= 0;
end
else begin
if(next_set)
    bram2_addr <= 0;

else if(acc_cnt == 9 && !latency)
 bram2_addr <= bram2_addr;
 
else if (wt_cnt < 27 && wt_cnt > 0 && start && !latency )
bram2_addr <= bram2_addr + ADDR_JUMP;

else if(!latency) bram2_addr <= 0;
end
end


// block for resettig the counter every 9 cycles for reading from input weights bram
always@ (posedge clk) begin

if (!reset) begin
wt_cnt <= 1;
end
else begin
    if(next_set)
        wt_cnt <= 1;

    else if(acc_cnt == 9 && !latency)
            wt_cnt <= wt_cnt;
   else if(wt_cnt == 27 && start && !latency)
     wt_cnt <= 1;
   else if(start && !latency) begin
    wt_cnt <= wt_cnt + 1;
end
end
end

// block for resetting the accumulator registers counter (control logic)
always@ (posedge clk) begin
if (!reset) begin
  acc_cnt <= 0;
end
    else begin
        if(next_set)
            acc_cnt <= 0;
      else if(acc_cnt == 10 && start && !latency)
          acc_cnt <= 1;
      else if(start && !latency)
          acc_cnt <= acc_cnt + 1;
    end
end

always @(posedge clk) begin
    if(!reset)begin
    s_axis_a_tvalid <= 0;
    s_axis_b_tvalid <= 0;
    s_axis_c_tvalid <= 0;
    end
    else if(next_set) begin
    s_axis_a_tvalid <= 0;
    s_axis_b_tvalid <= 0;
    s_axis_c_tvalid <= 0; 
    end
    else if(start && !latency && !flag1)begin
    s_axis_a_tvalid <= 1;
    s_axis_b_tvalid <= 1;
    s_axis_c_tvalid <= 1;
    end
    else begin
    s_axis_a_tvalid <= 0;
    s_axis_b_tvalid <= 0;
    s_axis_c_tvalid <= 0;
    end

end

//end of bram1 addresses
always@(posedge clk) begin
    if(!reset)
        flag1 <= 0;
    else if(next_set)
        flag1 <= 0;
     else if(bram1_addr == 3072*ADDR_JUMP)
        flag1 <= 1;
end

// block for updating the accumulator register -- works for 1
always @ (posedge clk) begin

if(!reset)
acc_reg <= 0;
else if(next_set)
    acc_reg <= 0;
else if ( acc_cnt < 10 && start && m_axis_result_tvalid)
 acc_reg <= m_axis_result_tdata;
else if ( acc_cnt < 10 && start)
 acc_reg <= acc_reg;
else acc_reg <= 0;
end


// block for writing data into output buffer
always @ (posedge clk) begin

if(!reset)
bram3_wrdata <= 0;

else
begin
if(next_set)
    bram3_wrdata <= 0;
else begin    
if(flag == 1 && start && !flag1)
begin
bram3_we <= 1;

    if(acc_reg[31] == 0)
    bram3_wrdata <= acc_reg;
    else
    bram3_wrdata <= 0;
end
else
 bram3_we <= 0;
end
end
end

// control block for setting read and write enables of depthwise output BRAM
always @ (posedge clk) begin

if(!reset)
bram3_addr_wr <= 0;

else begin
    if(next_set)
        bram3_addr_wr <= 0;

 else begin
  if(acc_cnt == 9 && start && !latency)
 flag <= 1;
  else
 flag <= 0;

  if(bram3_we == 1 && start)
 bram3_re <= 1;
 else if(flag1 && latency==LATENCY)
 bram3_re <= 1;
  else
 bram3_re <= 0;

  if(bram3_re == 1 && start)
  bram3_addr_wr <= bram3_addr_wr + 1;
end
end
end

endmodule

//pointwise module

module pointwise (

parameter T=32;
input clk,
input reset,
input bram3_re,
input bram3_we,
input       [T-1:0]  start, next_set,
input       [T-1:0] bram3_rddata,
input       [T-1:0] bram4_rddata,
input        pw_m_axis_result_tvalid,
input       [T-1:0] pw_m_axis_result_tdata,
output reg  [3:0]  bram5_we,
output reg  [T-1:0]  bram4_addr,  
output reg  [T-1:0] bram5_addr,    
output reg  [12:0] bram3_addr_rd,
output reg  [T-1:0] stop,
output reg  [T-1:0] bram5_wrdata,
output    pw_s_axis_a_tvalid,
    output    pw_s_axis_b_tvalid,
    output    pw_s_axis_c_tvalid,
    output reg [T-1:0] temp);

parameter ADDR_JUMP = 1;
parameter LATENCY = 1;
    reg [3 : 0] latency;
reg [4:0]  pw_wt_cnt;    
reg    r_en, g_en, b_en;
reg [11:0]  val7_addr;
reg flag2, flag3, flag4;
reg [T-1:0] r, g, b;


//latency control
always@(posedge clk) begin
if(!reset)
    latency<=0;
else begin
if (next_set)
    latency<=0;
else if (start)
begin
    if(latency == LATENCY)
        latency <= 0;
    else
        latency<= latency+1;
    end
end
end

//block for pointwise input buffer
always @(posedge clk) begin
if(!reset)
    bram3_addr_rd <= 0;

else
begin
if (next_set)
    bram3_addr_rd <= 0;

else if(pw_wt_cnt < 9 && start && bram3_re)
begin
if(pw_wt_cnt %3 == 0)
    bram3_addr_rd <= val7_addr + 1;
else if(pw_wt_cnt %3 == 2)
    bram3_addr_rd <= val7_addr;    
else
    bram3_addr_rd <= bram3_addr_rd + 1;
end
end // end reset else
end //end posedge block

//block for pw weight buffer
always @ (posedge clk) begin
if(!reset) begin
    bram4_addr <= 0;
    pw_wt_cnt <= 0;
    val7_addr <= 0;
end

else
begin
if(next_set) begin
    bram4_addr <= 0;
    pw_wt_cnt <= 0;
    val7_addr <= 0;  
    end
else begin     
    if(pw_wt_cnt == 7 && bram3_re)
        val7_addr <= val7_addr + 3;
    
    if(start && bram3_re)
    begin
        if(pw_wt_cnt == 8)
        begin
            bram4_addr <= 0;
            pw_wt_cnt <= 0;
        end
        else
        begin
            bram4_addr <= bram4_addr + ADDR_JUMP;
            pw_wt_cnt <= pw_wt_cnt + 1;
        end
    end
end //end next_set 
end // reset end
end// end posedge block


assign pw_s_axis_a_tvalid = (r_en || g_en || b_en) ? 1 : 0;
assign pw_s_axis_b_tvalid = (r_en || g_en || b_en) ? 1 : 0;
assign pw_s_axis_c_tvalid = (r_en || g_en || b_en) ? 1 : 0;


//control logic for updating r,g,b control signals
always @(posedge clk) begin

if(!reset)
begin
r_en <= 0;
g_en <= 0;
b_en <= 0;
temp <= 0;
flag3 <= 0;
end
else
begin
if(next_set) begin
    r_en <= 0;
    g_en <= 0;
    b_en <= 0;
    temp <= 0;
    flag3 <= 0; 
    end

else begin
if(bram3_re)
begin

if(pw_wt_cnt %3 == 0)
begin
    r_en <= 1;
    g_en <= 0;
    b_en <= 0;

end

else if(pw_wt_cnt %3 == 1)
begin
    r_en <= 0;
    g_en <= 1;
    b_en <= 0;
end

else if(pw_wt_cnt %3 == 2)
begin
    r_en <= 0;
    g_en <= 0;
    b_en <= 1;
end

end
else begin
    r_en <= 0;
    g_en <= 0;
    b_en <= 0;
end

if( pw_m_axis_result_tvalid)
begin
    temp <= pw_m_axis_result_tdata;
end

if (b_en)
    flag3 <= 1;

//0 change
if (LATENCY==0 && flag3) begin
            flag3 <= 0;    
            temp <= 0;
            end

else if (flag3 && pw_m_axis_result_tvalid) begin
        flag3 <= 0;    
        temp <= 0;
        end    
end
end// end reset 
end// end control logic block

//block for writing data into output buffer
always @ (posedge clk) begin

if(!reset)
begin
    bram5_wrdata <= 0;
    bram5_addr <= 0;
    flag4 <= 0;
    stop <= 0;
end

else
begin
if(next_set) begin
    bram5_wrdata <= 0;
    bram5_addr <= 0;
    flag4 <= 0;
    stop <= 0;
end
else begin
if(flag3 && !stop && LATENCY == 0)
    begin
        if(temp[31] == 0)
        begin
                bram5_wrdata <= temp;            
                bram5_we <= 4'hf;
                flag4 <= 1;
        end
        else
                bram5_wrdata <= 0;
    end

else if(flag3 && !stop &&pw_m_axis_result_tvalid)
begin
if(temp[31] == 0)
begin

    bram5_wrdata <= pw_m_axis_result_tdata; //begin        
    bram5_we <= 4'hf;
    flag4 <= 1;
                   // end
end
else
bram5_wrdata <= 0;
end

else
bram5_we <= 0;

if(flag4 && !stop) begin
bram5_addr <= bram5_addr + ADDR_JUMP;
flag4 <= 0;
if(bram5_addr == 2699*ADDR_JUMP)
    stop <= 32'h0000ffff;
   
        end
        end
end // end reset else
end//end posedge block

endmodule

// BRAM required
module bram3(

input clk,
input reset,
input  bram3_we,
input  bram3_re,
input  [31:0] bram3_wrdata,
input  [12:0] bram3_addr_wr,
input  [12:0] bram3_addr_rd,
output reg [31:0] bram3_rddata);

reg [31:0] mem [2699:0];

always @(posedge clk) begin

if(!reset)
bram3_rddata <= 0;
else
begin

if(bram3_we)
mem[bram3_addr_wr] <= bram3_wrdata;

if(bram3_re)
bram3_rddata <= mem[bram3_addr_rd];
end
end

endmodule

