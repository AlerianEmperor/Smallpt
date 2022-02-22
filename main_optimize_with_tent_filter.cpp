#include <iostream>
#include <fstream>
#include <omp.h>
#include <time.h>
#include <string>
#include <vector>
#include "fasttrigo.h"
#define eps 1e-4
#define pi 3.1415926
#define ipi 1.0f / pi
#define Width 1024
#define Height 768
#define iWidth 1.0 / Width
#define iHeight 1.0 / Height
////#define abs(x) ((x)<0 ? -(x) : (x))//default abs is faster than macro abs
//full optimze give 3x speed up( using faster sphere intersection and FTA)
//partly optimize give 2.5x speed up
using namespace std;

enum material { Diffuse, Glass, Mirror, Light };

double inline __declspec (naked) __fastcall sqrt14(double n)
{
	_asm fld qword ptr[esp + 4]
		_asm fsqrt
	_asm ret 8

}
thread_local uint32_t s_RndState = 1;
static const double imax = 1.0 / UINT32_MAX;
static const double irand_max = 1.0 / RAND_MAX;
static double randf()
{
	uint32_t x = s_RndState;
	x ^= x << 13;
	x ^= x >> 17;
	x ^= x << 15;
	s_RndState = x;
	return x * imax;
}
struct vec3
{
	vec3(){}
	vec3(double v) : x(v), y(v), z(v) {}
	vec3(double x_ , double y_ , double z_) : x(x_), y(y_), z(z_) {}
	double x, y, z;
	friend vec3 operator+(const vec3& a, const vec3& b) { return{ a.x + b.x, a.y + b.y, a.z + b.z }; }
	friend vec3 operator-(const vec3& a, const vec3& b) { return{ a.x - b.x, a.y - b.y, a.z - b.z }; }
	friend vec3 operator*(const vec3& a, const vec3& b) { return{ a.x * b.x, a.y * b.y, a.z * b.z }; }
	vec3 operator*=(const vec3& v) { return{ x *= v.x, y *= v.y, z *= v.z }; }
	friend vec3 operator-(const vec3& a) { return{ -a.x, -a.y, -a.z }; }
	vec3 __fastcall  operator+=(const vec3& v) { x += v.x; y += v.y; z += v.z; return *this; }
	vec3 __fastcall  operator*=(const double& value) { x *= value; y *= value; z *= value; return *this; }
	vec3 __fastcall norm() const { const double l = 1.0 / sqrt14(x*x + y*y + z*z); return *this * l; }
	double __fastcall dot(const vec3& v) const { return x * v.x + y * v.y + z * v.z; }
	vec3 __fastcall  cross(const vec3& v) const { return vec3(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x); }
};
struct Ray
{
	Ray(vec3 o_, vec3 d_) : o(o_), d(d_) {}
	vec3 o, d;
};
struct Sphere
{
	double rad;
	vec3 p, c, e;
	int mtl_type = Diffuse;
	Sphere(double rad_, vec3 p_, vec3 c_, vec3 e_, int mtl_type_) : rad(rad_), p(p_), c(c_), e(e_), mtl_type(mtl_type_) {}
	bool __fastcall intersect(const Ray& r, double& t) const
	{
		//-------------ORIGINAL VERSION-------------
		//-------------SLOWEST VERSION-----------------
		/*vec3 oc(r.o - p);
		double a = r.d.dot(r.d), b = oc.dot(r.d), c = oc.dot(oc) - rad * rad;
		double discriminant = b*b - a*c;
		if (discriminant < 0.0)
			return false;
		else
		{
			double dis = sqrt14(discriminant);	
			double inv_a = 1.0 / a;
			double tmin = (-b - dis) * inv_a;
			if (tmin > eps)
			{
				t = tmin;
				return true;
			}
			double tmax = (-b + dis) * inv_a;
			if (tmax > eps)
			{
				t = tmax;
				return true;
			}
			return false;
		}
		*/	
		//-------------------SMALLPT VERSION----------------
		//---------------SECOND FASTEST- FASTER THAN ORIGINAL VERSION - SLOWER THAN OPTIMIZE VERSION
		/*vec3 op(p - r.o); // Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0 
		double b = op.dot(r.d), det = b*b - op.dot(op) + rad*rad;
		if (det<0) return false; else det = sqrt(det);
		return (t = b - det)>eps ? true : ((t = b + det)>eps ? true : false);*/

		//--------------------OPTIMIZE VERSION------------
		//--------------------FASTEST VERSION-----------------
		//double a = r.d.dot(r.d)
		//but r.d is 1 because ray are normalize
		//so a is not needed
		vec3 oc(r.o - p);
		double b = oc.dot(r.d), c = oc.dot(oc) - rad * rad;
		double discriminant = b*b - c;
		if (discriminant < 0.0)
			return false;
		else
		{
			double dis = sqrt14(discriminant);

			double tmin = (-b - dis);// *inv_a;
			if (tmin > eps)
			{
				t = tmin;
				return true;
			}
			double tmax = (-b + dis);// *inv_a;
			if (tmax > eps)
			{
				t = tmax;
				return true;
			}
			return false;
		}
	}
};
double a = 12.0;
const Sphere spheres[] = {//Scene: radius, position, color, emission, material
	Sphere(1e5, vec3(-1e5 + 1.0,40.8, 81.6), vec3(.75, .25, .25), vec3(),Diffuse),//Left
	Sphere(1e5, vec3(1e5 + 99.0, 40.8, 81.6), vec3(.25, .25, .75), vec3(), Diffuse),//Rght
	Sphere(1e5, vec3(50.0, 40.8, -1e5), vec3(.75, .75, .75),  vec3(), Diffuse),//Back
	//Sphere(1e5, vec3(50.0, 40.8, 1e5 + 170.0), vec3(),vec3(),  Diffuse),//Frnt
	Sphere(1e5, vec3(50.0, -1e5, 81.6), vec3(0.75, .75 ,.75), vec3(), Diffuse),//Botm
	Sphere(1e5, vec3(50.0, 1e5 + 81.6, 81.6), vec3(.75, .75, .75), vec3(), Diffuse),//Top
	Sphere(16.5, vec3(27.0,16.5, 47.0), vec3(1.0, 1.0, 1.0) * .999, vec3(), Mirror),//Mirr
	Sphere(16.5, vec3(73.0, 16.5,78.0), vec3(1.0, 1.0, 1.0) * 0.999, vec3(), Glass),//Glas
	Sphere(600.0, vec3(50.0, 681.6 - 0.27 ,81.6), vec3(0.0, 0.0, 0.0), vec3(a, a, a), Light) //Lite
};
static void onb(vec3& n, vec3& u, vec3& v)
{
	if (n.z >= -0.9999999)// Handle the singularity
	{ 
		const double a = 1.0 / (1.0 + n.z);
		const double b = -n.x * n.y * a;
		u = vec3(1.0 - n.x * n.x * a, b, -n.x);
		v = vec3(b, 1.0 - n.y * n.y * a, -n.y);
		return;
	}
	u = vec3(0.0, -1.0, 0.0);
	v = vec3(-1.0, 0.0, 0.0);
	return;
}
double clampf(const double& x) { return x < 0 ? 0 : x > 1 ? 1 : x; }
int toInt(const double& x) { return int(pow(clampf(x), 1 / 2.2) * 255 + 0.5); }
static bool intersect(const Ray& r, const int& n, double& t, int& id)
{	
	t = 1e20;
	for (int i = n - 1; i >= 0; --i)
	{
		double d;
		bool hit = spheres[i].intersect(r, d);
		if (hit && d <= t && d >= eps)
		{
			t = d; id = i;
		}
	}
	return t < 1e20;
}
static vec3 Radiance(const Ray& r, const int& num_spheres)
{
	Ray new_ray(r);
	vec3 L(0.0), T(1.0);

	for (int z = 0; z <= 10; ++z)
	{
		double t = 1e20;
		int id;
		if (intersect(new_ray, num_spheres, t, id))
		{
			vec3 hit_point = new_ray.o + new_ray.d * t;

			vec3 normal = (hit_point - spheres[id].p).norm();

			if (spheres[id].mtl_type == Light)
			{
				L += T * spheres[id].e;
				return L;
			}
			else if (spheres[id].mtl_type == Diffuse)
			{
				double r1 = randf(), r2 = randf(), r2s = sqrt14(r1);
				double theta = 2.0 * pi * r2;

				//cosf, sinf is much slower than FTA::sincos
				//double c = cosf(theta), s = sinf(theta);
				double s, c;
				FTA::sincos(theta, &s, &c);
				vec3 u, v;
				onb(normal, u, v);

				vec3 direction_out(r2s * (u * c + v * s) + normal * sqrt14(1.0 - r1));

				new_ray = Ray(hit_point + 0.02 * normal, direction_out);

				T *= spheres[id].c;
			}
			else if (spheres[id].mtl_type == Mirror)
				new_ray = Ray(hit_point + 0.02 * normal, (new_ray.d - 2.0 * new_ray.d.dot(normal) * normal).norm());
			else if (spheres[id].mtl_type == Glass)
			{
				double cos_i = new_ray.d.dot(normal);
				const double ior = 1.6;
				double eta = cos_i >= 0.0 ? ior : 1.0 / ior;

				vec3 real_normal = cos_i >= 0.0 ? -normal : normal;

				double c2 = 1.0 - eta * eta * (1.0 - cos_i * cos_i);

				double R;

				vec3 Refractive_direction;

				if (c2 < 0.0)
					R = 1.0;
				else
				{
					double abs_cos_i = abs(cos_i);
					Refractive_direction = (eta * new_ray.d + (eta * abs_cos_i - sqrt14(c2)) * real_normal).norm();
					double cos_t = abs(Refractive_direction.dot(normal));

					double R0 = (ior - 1.0) / (ior + 1.0);

					R0 = R0 * R0;

					double p = 1.0 - cos_t, p2 = p * p;

					R = R0 + (1.0 - R0) * p2 * p2 * p;
				}

				double Tr = 1.0 - R;

				//Realistic Ray Tracing 177 and 178
				//page 178
				//P = k / 2 + (1 - k) * R
				//here k = 0.5

				double P = 0.25 + 0.5 * R;

				if (randf() < R)
				{
					new_ray = Ray(hit_point + 0.2 * real_normal, (new_ray.d - 2.0 * new_ray.d.dot(real_normal) * real_normal).norm());
					T *= (R / P);
				}
				else
				{
					new_ray = Ray(hit_point - 0.2 * real_normal, Refractive_direction);
					T *= (1.0 - R) / (1.0 - P);
				}
			}
			if (z >= 5)
			{
				vec3 f = spheres[id].c;
				double p = f.x > f.y && f.x > f.z ? f.x : f.y > f.z ? f.y : f.z;

				if (randf() <= p)
					T *= (1.0 / p);
				else
					return L;
			}
		}
		else
			return (L + T);//vec3(0, 0, 0);
	}
	return vec3(0.0);
}

void main()
{
	int ns = 40;
	ns /= 4;
	const float ins = 1.0 / ns;

	vec3 origin(50.0, 50.0, 295.6);
	
	vec3 d(0, -0.042612, -1);//vec3 d(0.0, 0.0, -1.0);

	d = d.norm();
	//origin += 150.0 * d;
	//origin += 140.0 * d;
	const vec3 w = -d;
	const vec3 up = vec3(0, 1, 0);
	vec3 u = up.cross(w).norm();
	vec3 v = w.cross(u);

	double tan_theta = tanf(54.0f);//tanf(40.0 * pi / 180.0);
	double aspect_ratio = double(Width) / double(Height);

	u = u * aspect_ratio * tan_theta;
	v = v * tan_theta;

	int num_spheres = sizeof(spheres) / sizeof(Sphere);

	vector<vec3> c(Width * Height);

	omp_set_num_threads(128);

	clock_t t_render = clock();

	for (int j = 0; j < Height; ++j)
	{
		fprintf(stderr, "\rRendering: (%d spp) %5.2f%%", 4 * ns, 100.0f * j / (Height - 1));
		#pragma omp parallel for schedule(guided)
		for (int i = 0; i < Width; ++i)
		{
			//vec3 color(0.0);
			
			/*for (int sx = 0; sx < 2; ++sx)
			{
				for (int sy = 0; sy < 2; ++sy)
				{
					for (int s = 0; s < ns; ++s)
					{
						double p = ((double)(i + sx / 2) + 0.5 * randf()) * iWidth;
						double q = ((double)(j + sy / 2) + 0.5 * randf()) * iHeight;

						p = (2.0 * p - 1.0) * aspect_ratio * tan_theta;
						q = (1.0 - 2.0 * q) * tan_theta;

						Ray r(origin, (u * p + v * q - w).norm());
						color += Radiance(r, num_spheres);
					}
				}
			}*/

			//Tent Filter from Realistic Rendering Peter Shirley page 60 
			//wrong
			/*for (int sx = 0; sx < 2; ++sx)
			{
				for (int sy = 0; sy < 2; ++sy)
				{
					for (int s = 0; s < ns; ++s)
					{
						double r1 = 2.0 * randf();
						double r2 = 2.0 * randf();
						
						r1 = r1 < 1 ? sqrt14(r1) - 1.0 : 1.0 - sqrt14(2.0 - r1);
						r2 = r2 < 1 ? sqrt14(r2) - 1.0 : 1.0 - sqrt14(2.0 - r2);

						double p = (double)(i + r1) * iWidth;
						double q = (double)(j + r2) * iHeight;

						p = (2.0 * p - 1.0) * aspect_ratio * tan_theta;
						q = (1.0 - 2.0 * q) * tan_theta;

						//r1 = (2.0 * r1 - 1.0) * aspect_ratio * tan_theta;
						//r2 = (1.0 - 2.0 * r2) * tan_theta;

						Ray r(origin, (u * p + v * q - w).norm());
						color += Radiance(r, num_spheres);
					}
				}
			}*/

			//Tent Filter from Realistic Rendering Peter Shirley page 60 
			vec3 sum(0.0f);
			for (int sx = 0; sx < 2; ++sx)
			{
				for (int sy = 0; sy < 2; ++sy)
				{				
					for (int s = 0; s < ns; ++s)
					{
						double r1 = 2.0 * randf();
						double r2 = 2.0 * randf();

						r1 = r1 < 1 ? sqrt14(r1) - 1.0 : 1.0 - sqrt14(2.0 - r1);
						r2 = r2 < 1 ? sqrt14(r2) - 1.0 : 1.0 - sqrt14(2.0 - r2);

						double p = ((sx + 0.5 + r1) / 2 + i) * iWidth - 0.5;
						double q = 0.5 - ((sy + 0.5 + r2) / 2 + j) * iHeight;
					
						Ray r(origin, (u * p + v * q - w).norm());
						sum = sum + Radiance(r, num_spheres);
					}
					c[j * Width + i] = sum * ins * 0.25;
				}
			}
			
			
		}
	}
	float render_time = ((double)clock() - t_render) / CLOCKS_PER_SEC;
	std::cout << "\nRendering time: " << render_time << "\n";

	string name = "Final_Optimize_Ten_Filter_sx_sy_sum_40_spp";

	ofstream ofs(name + ".ppm");
	
	ofs << "P3\n" << Width << " " << Height << "\n255\n";
	
	clock_t t_write = clock();

	for (int i = 0; i < Width * Height; ++i)
		ofs << toInt(c[i].x) << " " << toInt(c[i].y) << " " << toInt(c[i].z) << "\n";

	vector<vec3>().swap(c);

	float write_time = ((double)clock() - t_write) / CLOCKS_PER_SEC;
	cout << "Write Time: " << write_time << "\n";

	ofstream ofs_log(name + "_Render_Log.txt");

	ofs_log << "Rendering time: " << render_time << "\n";
	ofs_log << "Write Time: " << write_time << "\n";
}