#pragma once
#include <Windows.h>	// for OutputDebugStringA
#include <memory>

namespace RS
{
	//configuration
#if defined(_DEBUG)
#define _RS_DEBUG_
#endif

	// for memory
#define rs_new  new
#define rs_delete delete
#define rs_delete_array delete[]

#define rs_inline inline

	// for type
	using uint = unsigned int;
	using uint32 = unsigned int;
	using uint64 = unsigned long long;

	// for string
#define STRINGFY(x) #x
#define STRING_APPEND(x, y) STRINGFY(x##y)

	// for log
#define MAX_LOG_BUFFER_LENGTH 32768
#define RS_LOG(developer, text, ...)													\
{																						\
	char buffer[MAX_LOG_BUFFER_LENGTH];													\
	sprintf_s(buffer, (MAX_LOG_BUFFER_LENGTH), "[" developer "]", text, ##__VA_ARGS__); \
	strcat_s(buffer, MAX_LOG_BUFFER_LENGTH, "\n");										\
	OutputDebugStringA(buffer);															\
}

	// for assert
#define RS_ASSERT_COMPILE(condition, msg) static_assert(condition, msg)
#define RS_ASSERT_DEV(develper, condition, text, ...)									\
{																						\
	if(!(condition))																	\
	{																					\
		RS_LOG(develper, text, ##__VA_ARGS__);											\
		__debugbreak();																	\
	}																					\
}

	// etc
	template<typename T, typename U>
	class pair
	{
	public:
		pair(const T& first, const T& second)
			: _first(first)
			, _second(second)
		{}

	public:
		T _first;
		U _second;
	};

	// RAII
	struct DefaultDeleter
	{
		template<typename T>
		rs_inline static void release(T*& p)
		{
			rs_delete(p);
		}
	};
	template<typename T, typename Deleter = DefaultDeleter>
	struct Ptr
	{
		Ptr()
			: _ptr(nullptr)
		{}
		Ptr(T* ptr)
			: _ptr(ptr)
		{}
		rs_inline ~Ptr()
		{
			release();
		}

		void assign(T* ptr)
		{
			release();
			_ptr = ptr;
		}

		void swap(Ptr<T>& rhs)
		{
			//_ptr.swap(rhs._ptr);
			std::swap(_ptr, rhs._ptr);
		}

		void release()
		{
			Deleter::release(_ptr);
		}

		T* get()
		{
			return _ptr;
		}
		const T* get() const
		{
			return _ptr;
		}

		const T* operator->() const { return _ptr; }
		T* operator->() { return _ptr; }

	private:
		T* _ptr = nullptr;
	};
}

// for lib
#include <vector>
template<typename T>
using vector = std::vector<T>;
#pragma warning(disable : 4267)	// #TODO- 나중에 vector 클래스를 wrapping하고 반드시 지워야함

// for math
#include "Math.h"