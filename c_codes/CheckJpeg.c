#include "iostream"
#include "opencv2/opencv.hpp"
#include <string>
#include <vector>
#include <io.h>
#include <fstream>
#include <direct.h>
#include <stdlib.h> 


using namespace std;
using namespace cv;

#pragma comment(lib,"opencv_world341.lib")

//遍历文件夹下  所有文件夹
void ReadDirPath(string basePath, vector<string>& dirList)
{
	//dirList.push_back(basePath);
	//文件句柄  
	long long  hFile = 0;
	//文件信息  
	struct _finddata_t fileinfo;
	string p;
	if ((hFile = _findfirst(p.assign(basePath).append("\\*").c_str(), &fileinfo)) != -1)
	{
		do
		{
			//如果是目录,迭代之  
			//如果不是,加入列表  
			if (strcmp(fileinfo.name, ".") == 0 || strcmp(fileinfo.name, "..") == 0)
			{
				continue;
			}
			else
			{
				if ((fileinfo.attrib & 0x10) == _A_SUBDIR)
				{
					string dir = p.assign(basePath).append("\\").append(fileinfo.name);
					dirList.push_back(dir);
					ReadDirPath(dir, dirList);
				}

			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
}

//编译文件夹下 所有文件（这里加了后缀过滤）
void ReadImagePath(string basePath, vector<string>& imageList)
{
	if (!imageList.empty())
	{
		imageList.clear();
	}
	vector<string> dirList;
	dirList.push_back(basePath);
	ReadDirPath(basePath, dirList);
	for (int i = 0; i < dirList.size(); i++)
	{
		long long  hFile = 0;
		//文件信息  
		struct _finddata_t fileinfo;
		string p;
		if ((hFile = _findfirst(p.assign(dirList[i]).append("\\*.*").c_str(), &fileinfo)) != -1)
		{
			do
			{
				if (strcmp(fileinfo.name, ".") == 0 || strcmp(fileinfo.name, "..") == 0)
				{
					continue;
				}
				else
				{
					string name = fileinfo.name;
					if (name.size()<5)
					{
						continue;
					}
					name = name.substr(name.size() - 4, name.size());
					if (name == ".jpg" || name == ".JPG" || name == ".png" || name == ".PNG" || name == "jpeg" || name == "JPEG")
					{
						string dir = p.assign(dirList[i]).append("\\").append(fileinfo.name);
						imageList.push_back(dir);
					}
				}
			} while (_findnext(hFile, &fileinfo) == 0);
			_findclose(hFile);
		}
	}
}

//检查jpeg
bool CheckJpeg(string file)
{
	if (file.empty())
	{
		return false;
	}
	ifstream in(file.c_str(), ios::in | ios::binary);
	if (!in.is_open())
	{
		cout << "Error opening file!" << endl;
		return false;
	}

	int start;
	in.read((char*)&start, 4);
	short int lstart = start << 16 >> 16;

	//cout << hex << lstart << "  ";
	in.seekg(-4, ios::end);
	int end;
	in.read((char*)&end, 4);
	short int lend = end >> 16;
	//cout << hex << lend << endl;

	in.close();
	if ((lstart != -9985) || (lend != -9729)) //0xd8ff 0xd9ff
	{
		return true;
	}
	return false;
}

int main(int argc, char* argv[])
{
	if (argc < 3)
	{
		cout << "Please Use: [image_path] [isdemo]" << endl;
		system("pause");
		return -1;
	}
	else if (argc >3)
	{
		cout << "Parameters too much" << endl;
		system("pause");
		return -1;
	}

	string image_path = argv[1];
	
	vector<string> image_list;
	ReadImagePath(image_path, image_list);

	if (image_list.size()<1)
	{
		cout << image_path << ": This path has no jpeg image!" << endl;
		system("pause");
		return -1;
	}

	int num = image_list.size();
	cout << "Check image plan: " << endl;
	for (size_t i = 0; i < num; i++)
	{
		printf("%d/%d\r", i, num), fflush(stdout);
		string save_dir = image_path;

		string name = image_list[i];
		name = name.substr(name.size() - 4, name.size());
		bool isJpg = false;
		if (name == ".jpg" || name == ".JPG" || name == "jpeg" || name == "JPEG")
		{
			isJpg = CheckJpeg(image_list[i]);
		}
		
		if (isJpg)
		{
			save_dir += "_false";
		}
		
		Mat img = imread(image_list[i]);
		if (atoi(argv[2]) && !img.empty())
		{
			imshow("img", img);
			cvWaitKey(0);
		}

		if (isJpg) //格式破损
		{
			if (_access(save_dir.c_str(), 6) == -1)
			{
				_mkdir(save_dir.c_str());
			}
			if (!img.empty())
			{
				string image_name = image_list[i];
				image_name = image_name.substr(image_name.rfind("\\"));
				imwrite(save_dir + image_name, img);
			}
			remove(image_list[i].c_str());
		}
	}
	cout << "finished!" << endl;
	//system("pause");
	return 0;
}

