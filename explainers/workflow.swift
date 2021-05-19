
import io;
import python;

foreach i in [0:2]
{
  msg = python_persist("import cf_fake as cf", "cf.run(%i)"%i);
// is this allowed? how to dump all msgs to a pickle file
//  python_persist("import pickle", pickle.dump(msg, open("save.p", "wb")))
  printf("python result: %s", msg);
}
