
import java.io.*;

/*
  Header is first MB
*/

public class Regularizer
{
  int lines = 1;
  String input;
  BufferedReader in;
  String output;
  // PrintWriter out;
  RandomAccessFile out;
  long chunkSize;
  long inputSize;
  
  public static void main(String[] args)
  {
    Regularizer R = new Regularizer();
    try
    {
      R.go(args);
    }
    catch (UserException e)
    {
      System.out.println(e);
      System.exit(1);
    }
    finally
    {
      R.flush();
    }
  }

  void go(String[] args)
    throws UserException
  {
    if (args.length != 3)
      throw new UserException("Requires 3 arguments: input output chunks");

    input  = args[0];
    output = args[1];

    try
    {
      chunkSize = Long.parseLong(args[2]);
    }
    catch (NumberFormatException e)
    {
      throw new UserException("Bad chunk size: " + args[2]);
    }
    regularize();
  }

  void regularize()
    throws UserException
  {
    try
    {
      inputSize = new File(input).length();
      in = new BufferedReader(new FileReader(input));
    }
    catch (IOException e)
    {
      throw new UserException("Could not read from: " + input);
    }
    try
    {
      new FileWriter(output).close();
      out =
        // new PrintWriter(new BufferedWriter(new FileWriter(output)));
        new RandomAccessFile(output, "rw");
    }
    catch (IOException e)
    {
      throw new UserException("Could not write to: " + output);
    }
    xfer();
  }

  void xfer()
    throws UserException
  {
    writeChunkHeader();
    String extra = null;
    int chunk = 1;
    do
    {
      extra = writeChunk(chunk,  extra);
      System.out.println("extra: " + extra);
      chunk++;
    }
    while (extra != null);
  }

  void writeChunkHeader()
    throws UserException
  {
    StringBuilder sb = new StringBuilder();
    sb.append("chunkSize: " + chunkSize + "\n");
    sb.append("inputSize: " + inputSize);
    if (sb.length() > chunkSize)
      throw new UserException
        ("Header too long for given chunk size (" + chunkSize + ")!");
    println(sb.toString());
  }

  String writeChunk(int chunk, String extra)
    throws UserException
  {
    try
    {
      out.seek(chunk * chunkSize);
      long written = 0;
      if (extra != null)
      {
        written += extra.length() + 1;
        if (written > chunkSize)
          throw new UserException("Line " + lines + " too long!");
        lines++;
      }
      String s;
      while ((s = in.readLine()) != null)
      {
        written += s.length() + 1;
        if (written > chunkSize)
          return s;
        lines++;
        println(s);
      }
    }
    catch (IOException e)
    {
      throw new UserException("Error copying data!");
    }
    return null;
  }

  void println(String s)
    throws UserException
  {
    try
    {
      out.write(s   .getBytes("UTF-8"));
      out.write("\n".getBytes("UTF-8"));
    }
    catch (IOException e)
    {
      throw new UserException("write failed!");
    }
  }
  
  void flush()
  {
    System.out.println("flush");
    // out.flush();
  }
}

class UserException extends Exception
{
  String msg;
  UserException(String msg) { this.msg = msg; }
  public String toString() { return msg; }
}
