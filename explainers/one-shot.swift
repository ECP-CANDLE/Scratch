
import io;
import python;

msg = python_persist("import cf_fake as cf", "cf.run()");
printf("python result: %s", msg);
