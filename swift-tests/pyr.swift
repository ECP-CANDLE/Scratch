
import io;
import python;
import R;
import sys;

printf("LD_LIBRARY_PATH: %s", getenv("LD_LIBRARY_PATH"));
printf("R_HOME: %s", getenv("R_HOME"));

i = python("print(\"python works\")",
           "repr(2+2)");
printf("i: %s", i);

printf(R("", "\"R STRING OK\""));
