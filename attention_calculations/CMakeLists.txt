add_executable(attention_calculations 
	"attention.h" 
	"matmul_type.h" 
	"random.h"

	"tensor.cpp"
	"attention_vectorized.cpp"
	"attention.cpp"

	"attention_calculations.cpp" 
 "random.cpp")

target_include_directories(attention_calculations PRIVATE .)