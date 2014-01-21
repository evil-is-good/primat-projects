/*
 * =====================================================================================
 *
 *       Filename:  file.desc.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  11.11.2013 09:57:16
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */

namespace prmt
{

class File
{
    public:
        File (const str name, const str mode) {
            F = fopen (name.c_str(), mode.c_str); is_open = true; };
        File (const str name) { this->name = name;};
        void read (std::function<void(const str)> doit)
        {
            if (is_open)
            {
                while (
            }
        }
    private:
        File () {};
        str name;
        bool is_open = false;
};

};
