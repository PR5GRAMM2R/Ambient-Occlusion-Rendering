#pragma once

namespace RS
{
	static constexpr const uint32 uint32_max = 0xffffffff;

	struct float2
	{
		float2()
			: _x(0)
			, _y(0)
		{
		}
		float2(const float x, const float y)
			: _x(x)
			, _y(y)
		{}

		union
		{
			struct
			{
				float _x;
				float _y;
			};
			float _m[2];
		};
	};
	struct float3
	{
		float3()
			: _x(0)
			, _y(0)
			, _z(0)
		{
		}
		float3(const float x, const float y, const float z)
			: _x(x)
			, _y(y)
			, _z(z)
		{}

		const float3 operator+(const float3& rhs) const
		{
			return float3(_x + rhs._x, _y + rhs._y, _z + rhs._z);
		}
		const float3 operator-(const float3& rhs) const
		{
			return float3(_x - rhs._x, _y - rhs._y, _z - rhs._z);
		}

		const float3 operator/(const float rhs) const
		{
			return float3(_x / rhs, _y / rhs, _z / rhs);
		}

		const float3 cross(const float3& rhs) const
		{
			return float3(_y * rhs._z - _z * rhs._y, _z * rhs._x - _x * rhs._z, _x * rhs._y - _y * rhs._x);
		}

		union
		{
			struct
			{
				float _x;
				float _y;
				float _z;
			};
			float _m[3];
		};
	};
	struct float4
	{
		float4(const float x, const float y, const float z, const float w)
			: _x(x)
			, _y(y)
			, _z(z)
			, _w(w)
		{}
		float4(const float3& xyz, const float w)
			: _x(xyz._x)
			, _y(xyz._y)
			, _z(xyz._z)
			, _w(w)
		{}

		void operator/=(const float rhs)
		{
			_x /= rhs;
			_y /= rhs;
			_z /= rhs;
			_w /= rhs;
		}

		union
		{
			struct
			{
				float _x;
				float _y;
				float _z;
				float _w;
			};
			float _m[4];
		};
	};
	using quaternion = float4;

	struct Transform
	{
		float3 _scale;
		quaternion _rotation;
		float3 _translation;
	};

	struct float4x4
	{
		static const float4x4 IDENTITY;

		union
		{
			struct
			{
				float _00, _01, _02, _03, _10, _11, _12, _13, _20, _21, _22, _23, _30, _31, _32, _33;
			};
			float _m[16];
			float _rows[4][4];
		};
	};

	namespace Math
	{
		template<typename T>
		rs_inline T getMax(const T lhs, const T rhs)
		{
			return lhs > rhs ? lhs : rhs;
		}
	}
}