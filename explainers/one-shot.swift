
import io;
import python;

msg = python_persist("import cf_fake as cf", "cf.run(0)");
printf("python result: %s", msg);
