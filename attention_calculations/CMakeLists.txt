add_executable(attention_calculations 
    "src/random.cpp"
	"src/tensor.cpp"

	"src/attention/naive_attention.cpp"
	"src/attention/vectorized_attention.cpp"
	"src/attention/cache_friendly_attention.cpp"
	"src/attention/attention.cpp"

	#"src/attention/cache_friendly_tiled_attention.cpp"
	"src/attention/tiled_attention.cpp"

	"src/attention_calculations.cpp" 
)

target_include_directories(attention_calculations PRIVATE "include")