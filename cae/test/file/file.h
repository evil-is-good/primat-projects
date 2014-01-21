void create_file(str &&file_name)
{
    FILE* F;
    F = fopen(file_name.c_str(),"w");
    fclose(F);
};

void append_in_file(str &&file_name, str &&s)
{
    FILE* F;
    F = fopen(file_name.c_str(),"a");
    fprintf(F, "%s", s.c_str());
    fclose(F);
};
