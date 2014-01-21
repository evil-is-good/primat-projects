
class MyC
{
    private:
        int conte;
    public:
        MyC();
        MyC(int in);
        int get();
};

MyC::MyC()
{
    conte = 0;
}

MyC::MyC(int in)
{
    conte = in;
}

int MyC::get()
{
    return conte;
}
