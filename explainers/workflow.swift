
import io;
import python;

foreach i in [0:10]
{
  msg = python_persist("import cf_fake as cf", "cf.run(%i)"%i);
  printf("python result: %s", msg);
}
