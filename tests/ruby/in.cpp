#include <iostream>
#include <ruby.h>

using namespace std;

int main(void)
{
  ruby_init();
  ruby_init_loadpath();
  int status;
  rb_load_protect(rb_str_new2("./test.rb"), 0, &status);
  if (status) {
    VALUE rbError = rb_funcall(rb_gv_get("$!"), rb_intern("message"), 0);
    cerr << StringValuePtr(rbError) << endl;
  };
  ruby_finalize();
  return status;
}
