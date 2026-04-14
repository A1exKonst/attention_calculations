add_executable(attention_calculations 
    "src/random.cpp"
	"src/tensor.cpp"
	"src/attention_vectorized.cpp"
	"src/attention.cpp"

	"src/attention_calculations.cpp" 
)

target_include_directories(attention_calculations PRIVATE "include")