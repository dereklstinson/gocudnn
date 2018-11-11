#Stride Tensor

If anyone is confused with what these are you are not alone.  After some playing around with it, I thought I would post some of my insights.  Note since I do most of my coding in Go I am going to use the syntax of Go.

First lets talk about an 1D array and you want to represent it with dims := []int{N,C,H,W}. The strides of dims are strides := []int32{(C*H*W),(H*W),(W),1}.

[code]
//GetValueNCHW returns value from array A with dims []int from the location n,c,h,w
func GetValueNCHW(A []float32, dims []int, n,c,h,w int) float32{

//Find Strides for A

   s:=make([]int,4)
   stride:=1

     for i:=4,i>=0;i--{
        s[i] = stride 
        stride*=dims[i]  
     }
  
  return A[(s[0]*n)+(s[1]*c)+(s[2]*h)+(s[3]*w)]
}
[/code]
 

 to be continued